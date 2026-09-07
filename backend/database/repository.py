"""SQLite persistence for missions, snapshots and the audit trail.

Snapshots belong to a mission, which is what makes post-flight analysis and
mission replay possible. Resetting the live simulator no longer destroys history:
missions are retained until explicitly deleted.

The schema is intentionally close to what a time-series store would use, so
swapping SQLite for TimescaleDB or PostgreSQL is a repository-level change and
nothing above it needs to move.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS missions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT    NOT NULL,
        profile       TEXT    NOT NULL,
        started_at    TEXT    NOT NULL,
        ended_at      TEXT,
        samples       INTEGER NOT NULL DEFAULT 0,
        ingestion     TEXT,
        summary       TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshots (
        mission_id  INTEGER NOT NULL,
        seq         INTEGER NOT NULL,
        timestamp   TEXT    NOT NULL,
        engine_hours REAL,
        health      REAL,
        fault       TEXT,
        payload     TEXT    NOT NULL,
        PRIMARY KEY (mission_id, seq)
    )
    """,
    'CREATE INDEX IF NOT EXISTS idx_snapshots_mission ON snapshots (mission_id, seq)',
    """
    CREATE TABLE IF NOT EXISTS audit (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        actor     TEXT NOT NULL,
        action    TEXT NOT NULL,
        detail    TEXT,
        outcome   TEXT
    )
    """,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, path: str | Path = 'data/trinetra.db') -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute('PRAGMA journal_mode=WAL')
            self._migrate()
            for statement in SCHEMA:
                self.conn.execute(statement)
            self.conn.commit()

    def _migrate(self) -> None:
        """Move a pre-mission ``snapshots`` table aside instead of dropping it.

        The v1 schema stored flat ``(timestamp, payload)`` rows with no mission
        association. Rather than destroying that data, it is renamed so the new
        schema can be created alongside it.
        """
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots'").fetchone()
        if row is None:
            return
        columns = {info[1] for info in self.conn.execute('PRAGMA table_info(snapshots)')}
        if 'mission_id' in columns:
            return
        suffix = 1
        while self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (f'snapshots_legacy_v{suffix}',)).fetchone() is not None:
            suffix += 1
        self.conn.execute(f'ALTER TABLE snapshots RENAME TO snapshots_legacy_v{suffix}')
        self.conn.commit()

    # ------------------------------------------------------------------ missions
    def start_mission(self, name: str, profile: str, ingestion: str) -> int:
        with self._lock:
            cursor = self.conn.execute(
                'INSERT INTO missions (name, profile, started_at, ingestion) VALUES (?,?,?,?)',
                (name, profile, _now(), ingestion))
            self.conn.commit()
            return int(cursor.lastrowid)

    def finish_mission(self, mission_id: int, summary: dict) -> None:
        with self._lock:
            self.conn.execute(
                'UPDATE missions SET ended_at = ?, summary = ? WHERE id = ? AND ended_at IS NULL',
                (_now(), json.dumps(summary), mission_id))
            self.conn.commit()

    def list_missions(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            'SELECT * FROM missions ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        return [self._mission_row(row) for row in rows]

    def mission(self, mission_id: int) -> dict | None:
        row = self.conn.execute('SELECT * FROM missions WHERE id = ?', (mission_id,)).fetchone()
        return self._mission_row(row) if row else None

    def delete_mission(self, mission_id: int) -> bool:
        with self._lock:
            cursor = self.conn.execute('DELETE FROM missions WHERE id = ?', (mission_id,))
            self.conn.execute('DELETE FROM snapshots WHERE mission_id = ?', (mission_id,))
            self.conn.commit()
            return cursor.rowcount > 0

    def delete_all_missions(self) -> int:
        with self._lock:
            cursor = self.conn.execute('DELETE FROM missions')
            self.conn.execute('DELETE FROM snapshots')
            self.conn.commit()
            return cursor.rowcount

    @staticmethod
    def _mission_row(row: sqlite3.Row) -> dict:
        data = dict(row)
        if data.get('summary'):
            try:
                data['summary'] = json.loads(data['summary'])
            except json.JSONDecodeError:
                data['summary'] = None
        return data

    # ----------------------------------------------------------------- snapshots
    def add_snapshot(self, mission_id: int, seq: int, snapshot: dict) -> None:
        telemetry = snapshot.get('telemetry', {})
        with self._lock:
            self.conn.execute(
                'INSERT OR REPLACE INTO snapshots '
                '(mission_id, seq, timestamp, engine_hours, health, fault, payload) '
                'VALUES (?,?,?,?,?,?,?)',
                (mission_id, seq, telemetry.get('timestamp', _now()),
                 telemetry.get('engine_hours'),
                 snapshot.get('digital_twin', {}).get('health_score'),
                 snapshot.get('prediction', {}).get('predicted_fault'),
                 json.dumps(snapshot)))
            self.conn.execute('UPDATE missions SET samples = ? WHERE id = ?', (seq, mission_id))
            self.conn.commit()

    def snapshots(self, mission_id: int, start: int = 0, limit: int = 300) -> list[dict]:
        rows = self.conn.execute(
            'SELECT payload FROM snapshots WHERE mission_id = ? AND seq >= ? '
            'ORDER BY seq LIMIT ?', (mission_id, start, limit)).fetchall()
        return [json.loads(row['payload']) for row in rows]

    def snapshot_index(self, mission_id: int, step: int = 1) -> list[dict]:
        """Lightweight timeline for the replay scrubber, without full payloads."""
        rows = self.conn.execute(
            'SELECT seq, timestamp, engine_hours, health, fault FROM snapshots '
            'WHERE mission_id = ? ORDER BY seq', (mission_id,)).fetchall()
        return [dict(row) for row in rows[::max(1, step)]]

    def snapshot_count(self, mission_id: int) -> int:
        row = self.conn.execute(
            'SELECT COUNT(*) AS n FROM snapshots WHERE mission_id = ?', (mission_id,)).fetchone()
        return int(row['n'])

    # --------------------------------------------------------------------- audit
    def audit(self, actor: str, action: str, detail: Any = None, outcome: str = 'ok') -> None:
        with self._lock:
            self.conn.execute(
                'INSERT INTO audit (timestamp, actor, action, detail, outcome) VALUES (?,?,?,?,?)',
                (_now(), actor, action,
                 json.dumps(detail) if detail is not None else None, outcome))
            self.conn.commit()

    def audit_trail(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            'SELECT * FROM audit ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        out: list[dict] = []
        for row in rows:
            record = dict(row)
            if record.get('detail'):
                try:
                    record['detail'] = json.loads(record['detail'])
                except json.JSONDecodeError:
                    pass
            out.append(record)
        return out

    def close(self) -> None:
        with self._lock:
            self.conn.close()
