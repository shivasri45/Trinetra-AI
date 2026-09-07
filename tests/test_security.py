"""Authentication, telemetry integrity, audit trail and persistence."""
from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi import HTTPException

from backend.database.repository import Repository
from backend.security import telemetry_security as security

FRAME = {'timestamp': '2026-01-01T00:00:00Z', 'rpm': 4870.2, 'cht': 168.4, 'egt': 701.1}


# ------------------------------------------------------------------ integrity
def test_signature_is_stable_and_verifies():
    first = security.sign(FRAME)
    second = security.sign(FRAME)
    assert first['signature'] == second['signature']
    assert first['algorithm'] == 'HMAC-SHA256'
    assert len(first['signature']) == 64
    assert security.verify(FRAME, first['signature'])


def test_any_modification_invalidates_the_signature():
    signature = security.sign(FRAME)['signature']
    for field, value in (('rpm', 4870.3), ('cht', 168.5), ('timestamp', 'x')):
        tampered = {**FRAME, field: value}
        assert not security.verify(tampered, signature), field


def test_adding_or_removing_a_field_invalidates_the_signature():
    signature = security.sign(FRAME)['signature']
    assert not security.verify({**FRAME, 'extra': 1}, signature)
    assert not security.verify({k: v for k, v in FRAME.items() if k != 'egt'}, signature)


def test_verification_rejects_empty_and_malformed_signatures():
    assert not security.verify(FRAME, '')
    assert not security.verify(FRAME, 'not-a-signature')
    assert not security.verify(FRAME, 'a' * 64)


def test_canonical_form_is_order_independent():
    reordered = dict(reversed(list(FRAME.items())))
    assert security.canonical(FRAME) == security.canonical(reordered)
    assert security.verify(reordered, security.sign(FRAME)['signature'])


# --------------------------------------------------------------------- guards
def test_api_key_guard_accepts_the_configured_key():
    assert security.require_api_key(security.settings.api_key).startswith('key:')


@pytest.mark.parametrize('supplied', [None, '', 'wrong-key', 'trinetra-dev-key '])
def test_api_key_guard_rejects_everything_else(supplied):
    with pytest.raises(HTTPException) as exc:
        security.require_api_key(supplied)
    assert exc.value.status_code == 401


def test_websocket_key_check_mirrors_the_rest_guard():
    assert security.check_api_key(security.settings.api_key)
    assert not security.check_api_key('wrong')
    assert not security.check_api_key(None)


def test_posture_reports_the_active_controls():
    posture = security.security_posture()
    assert posture['authentication'] == 'api-key'
    assert posture['telemetry_integrity'] == 'HMAC-SHA256 per frame'
    assert posture['audit_logging'] == 'enabled'
    assert 'transport_encryption' in posture


def test_default_credentials_are_flagged(monkeypatch):
    # Settings is frozen by design, so swap in a replaced copy rather than mutating.
    monkeypatch.setattr(security, 'settings',
                        replace(security.settings, api_key=security.DEV_API_KEY))
    warnings = security.startup_warnings()
    assert any('default development API key' in w for w in warnings)


def test_disabled_authentication_is_flagged(monkeypatch):
    monkeypatch.setattr(security, 'settings', replace(security.settings, auth_enabled=False))
    assert any('Authentication is DISABLED' in w for w in security.startup_warnings())
    assert security.check_api_key(None), 'guard should pass through when disabled'


def test_production_refuses_default_credentials(monkeypatch):
    monkeypatch.setenv('TRINETRA_ENV', 'production')
    monkeypatch.setattr(security, 'settings',
                        replace(security.settings, api_key=security.DEV_API_KEY))
    with pytest.raises(RuntimeError, match='Refusing to start in production'):
        security.startup_warnings()


def test_production_refuses_disabled_authentication(monkeypatch):
    monkeypatch.setenv('TRINETRA_ENV', 'production')
    monkeypatch.setattr(security, 'settings', replace(security.settings, auth_enabled=False))
    with pytest.raises(RuntimeError):
        security.startup_warnings()


# ---------------------------------------------------------------- persistence
def test_mission_lifecycle_is_recorded(tmp_db_path):
    db = Repository(tmp_db_path)
    try:
        mission_id = db.start_mission('test sortie', 'normal_cruise', 'direct')
        for seq in range(1, 6):
            db.add_snapshot(mission_id, seq, {
                'telemetry': {'timestamp': f'2026-01-01T00:00:0{seq}Z', 'engine_hours': seq * 0.5},
                'digital_twin': {'health_score': 100 - seq},
                'prediction': {'predicted_fault': 'normal'},
            })
        assert db.snapshot_count(mission_id) == 5
        db.finish_mission(mission_id, {'samples': 5, 'health_min': 95})

        mission = db.mission(mission_id)
        assert mission['ended_at'] is not None
        assert mission['summary']['samples'] == 5
        assert mission['samples'] == 5
        assert len(db.snapshots(mission_id)) == 5
        assert len(db.snapshot_index(mission_id)) == 5
    finally:
        db.close()


def test_snapshot_paging_and_decimation(tmp_db_path):
    db = Repository(tmp_db_path)
    try:
        mission_id = db.start_mission('paging', 'normal_cruise', 'direct')
        for seq in range(1, 41):
            db.add_snapshot(mission_id, seq, {
                'telemetry': {'timestamp': 't', 'engine_hours': seq},
                'digital_twin': {'health_score': 90},
                'prediction': {'predicted_fault': 'normal'},
            })
        page = db.snapshots(mission_id, start=20, limit=5)
        assert len(page) == 5
        assert page[0]['telemetry']['engine_hours'] == 20
        assert len(db.snapshot_index(mission_id, step=4)) == 10
    finally:
        db.close()


def test_deleting_a_mission_removes_its_snapshots(tmp_db_path):
    db = Repository(tmp_db_path)
    try:
        mission_id = db.start_mission('doomed', 'hot_weather', 'can')
        db.add_snapshot(mission_id, 1, {'telemetry': {'timestamp': 't'},
                                        'digital_twin': {'health_score': 50},
                                        'prediction': {'predicted_fault': 'normal'}})
        assert db.delete_mission(mission_id)
        assert db.mission(mission_id) is None
        assert db.snapshot_count(mission_id) == 0
        assert not db.delete_mission(mission_id)
    finally:
        db.close()


def test_audit_trail_records_actor_action_and_outcome(tmp_db_path):
    db = Repository(tmp_db_path)
    try:
        db.audit('key:abc', 'simulation.start', {'mission_id': 1})
        db.audit('key:abc', 'mission.purge', {'removed': 3}, outcome='destructive')
        entries = db.audit_trail(10)
        assert len(entries) == 2
        assert entries[0]['action'] == 'mission.purge'
        assert entries[0]['outcome'] == 'destructive'
        assert entries[0]['detail'] == {'removed': 3}
        assert entries[1]['actor'] == 'key:abc'
    finally:
        db.close()


def test_legacy_schema_is_preserved_not_dropped(tmp_db_path):
    """A v1 database must be migrated without losing its rows."""
    import sqlite3
    connection = sqlite3.connect(tmp_db_path)
    connection.execute('CREATE TABLE snapshots (timestamp TEXT, payload TEXT)')
    connection.execute("INSERT INTO snapshots VALUES ('t', '{}')")
    connection.commit()
    connection.close()

    db = Repository(tmp_db_path)
    try:
        rows = db.conn.execute('SELECT COUNT(*) FROM snapshots_legacy_v1').fetchone()[0]
        assert rows == 1, 'legacy rows should be moved aside, not deleted'
        columns = {info[1] for info in db.conn.execute('PRAGMA table_info(snapshots)')}
        assert 'mission_id' in columns
    finally:
        db.close()
