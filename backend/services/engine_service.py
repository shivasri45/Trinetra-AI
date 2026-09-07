"""Orchestration: ingestion -> twin -> diagnostics -> RUL -> explanation -> fan-out.

``tick()`` is the single place where one acquisition cycle is turned into one
snapshot. The ground truth carried alongside a simulated frame is attached to the
snapshot only for demonstration and scoring, and is never passed into the twin,
the feature builder or the models.
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone

import numpy as np

from backend.config.settings import FAULTS, MISSION_PROFILES, settings
from backend.database.repository import Repository
from backend.digital_twin.synchroniser import TwinSynchroniser
from backend.ingestion.sources import build_source
from backend.ml.diagnostics import Diagnostics
from backend.ml.rul import RulEstimator
from backend.security import telemetry_security as security
from backend.signal_processing import spectral
from backend.simulator.engine import EngineSimulator
from backend.xai.explainer import Explainer


class EngineService:
    def __init__(self, repository: Repository | None = None) -> None:
        self.sim = EngineSimulator()
        self.db = repository or Repository(settings.db_path)
        self.ingestion_mode = settings.ingestion_mode
        self.source = build_source(self.ingestion_mode, self.sim)
        self.sync = TwinSynchroniser()
        self.ml = Diagnostics()
        self.rul = RulEstimator()
        self.explainer = Explainer(self.ml)

        self.latest: dict | None = None
        self.history: deque[dict] = deque(maxlen=settings.history_limit)
        self.spectrum: list[dict] = []
        self.raw_frames: list[dict] = []
        self.task: asyncio.Task | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.mission_id: int | None = None
        self.sequence = 0
        self.alarm_log: deque[dict] = deque(maxlen=200)
        self._last_fault = 'normal'

    # ------------------------------------------------------------------- config
    def set_ingestion(self, mode: str) -> str:
        if mode not in ('direct', 'can'):
            raise ValueError('ingestion mode must be "direct" or "can"')
        if mode != self.ingestion_mode:
            self.source.close()
            self.ingestion_mode = mode
            self.source = build_source(mode, self.sim)
        return self.ingestion_mode

    def configure_mission(self, mission: str) -> str:
        if mission not in MISSION_PROFILES:
            raise ValueError(f'unsupported mission profile: {mission}')
        self.sim.configure_mission(mission)
        return mission

    def inject_fault(self, fault: str) -> str:
        if fault not in FAULTS:
            raise ValueError(f'unsupported fault: {fault}')
        self.sim.inject_fault(fault)
        return fault

    # --------------------------------------------------------------------- tick
    def tick(self) -> dict:
        frame = self.source.read()
        telemetry = frame.telemetry
        result = self.sync.process(telemetry, frame.cylinders, frame.vibration)
        prediction = self.ml.analyse(result.features, result.twin)
        rul = self.rul.update(telemetry.get('engine_hours', 0.0), result.twin)
        explanation = self.explainer.explain(result.features, result.twin, prediction)

        if frame.waveform is not None:
            self.spectrum = spectral.spectrum_for_display(
                frame.waveform, settings.vib_sample_rate_hz)
        self.raw_frames = frame.raw_frames

        self.sequence += 1
        snapshot = {
            'sequence': self.sequence,
            'mission_id': self.mission_id,
            'mission_profile': self.sim.mission,
            'received_at': datetime.now(timezone.utc).isoformat(),
            'telemetry': telemetry,
            'cylinders': frame.cylinders,
            'vibration': frame.vibration,
            'digital_twin': result.twin,
            'prediction': prediction,
            'rul': rul,
            'explanation': explanation,
            'advisory': self.advisory(result.twin, prediction, rul),
            'ingestion': {'source': frame.source, **frame.transport},
            'integrity': security.sign(telemetry),
            # Ground truth for demonstration only; not an input to any analytic.
            'simulated_truth': frame.truth,
        }

        self.latest = snapshot
        self.history.append(snapshot)
        self._record_alarm(snapshot)
        if self.mission_id is not None:
            self.db.add_snapshot(self.mission_id, self.sequence, snapshot)
        for queue in list(self.subscribers):
            if not queue.full():
                queue.put_nowait(snapshot)
        return snapshot

    def _record_alarm(self, snapshot: dict) -> None:
        fault = snapshot['prediction']['predicted_fault']
        if fault != self._last_fault:
            self.alarm_log.append({
                'sequence': snapshot['sequence'],
                'timestamp': snapshot['telemetry'].get('timestamp'),
                'engine_hours': snapshot['telemetry'].get('engine_hours'),
                'from': self._last_fault,
                'to': fault,
                'health': snapshot['digital_twin']['health_score'],
                'confidence': snapshot['prediction']['confidence'],
                'triggers': snapshot['prediction']['detection_triggers'],
            })
            self._last_fault = fault

    # ------------------------------------------------------------------ advisory
    @staticmethod
    def advisory(twin: dict, prediction: dict, rul: dict) -> dict:
        health = twin['health_score']
        fault = prediction['predicted_fault']
        maintenance = rul['advisory']

        if health > settings.warning_health and fault == 'normal':
            dispatch, text = 'GO', 'Engine within limits. Cleared for extended-endurance dispatch.'
        elif health > settings.critical_health:
            dispatch = 'GO_WITH_LIMITATION'
            text = (f'Suspected {fault.replace("_", " ")}. Continue mission with increased '
                    f'monitoring; avoid sustained high power and plan inspection.')
        else:
            dispatch = 'NO_GO'
            text = (f'Suspected {fault.replace("_", " ")} with health at {health}. '
                    f'Do not dispatch on a long-endurance sortie; rectify before next flight.')

        return {
            'dispatch': dispatch,
            'summary': text,
            'maintenance_priority': maintenance['priority'],
            'maintenance_action': maintenance['action'],
            'due_in_hours': maintenance.get('due_hours'),
            'limiting_subsystem': rul['limiting_subsystem'],
        }

    # ------------------------------------------------------------------ mission
    def start(self, actor: str = 'system', name: str | None = None) -> dict:
        if self.mission_id is None:
            label = name or f'{self.sim.mission} {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}'
            self.mission_id = self.db.start_mission(label, self.sim.mission, self.ingestion_mode)
            self.sequence = 0
        self.sim.running = True
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._loop())
        self.db.audit(actor, 'simulation.start', {'mission_id': self.mission_id})
        return {'status': 'started', 'mission_id': self.mission_id, 'profile': self.sim.mission}

    def stop(self, actor: str = 'system') -> dict:
        self.sim.running = False
        mission_id = self.mission_id
        if mission_id is not None:
            self.db.finish_mission(mission_id, self.mission_summary())
            self.mission_id = None
        self.db.audit(actor, 'simulation.stop', {'mission_id': mission_id})
        return {'status': 'stopped', 'mission_id': mission_id}

    def reset(self, actor: str = 'system') -> dict:
        """Reset the live engine and analytics. Stored missions are preserved."""
        self.stop(actor)
        self.sim.reset()
        self.sync.reset()
        self.rul.reset()
        self.ml = Diagnostics()
        self.explainer = Explainer(self.ml)
        self.history.clear()
        self.alarm_log.clear()
        self.spectrum = []
        self.latest = None
        self.sequence = 0
        self._last_fault = 'normal'
        self.db.audit(actor, 'simulation.reset', {'missions_preserved': True})
        return {'status': 'reset', 'missions_preserved': True}

    async def _loop(self) -> None:
        while self.sim.running:
            try:
                self.tick()
            except Exception as exc:  # keep the loop alive, record the failure
                self.db.audit('system', 'tick.error', {'error': str(exc)}, outcome='error')
            await asyncio.sleep(settings.sample_period_s)

    def snapshot(self) -> dict:
        """Latest snapshot, producing one on demand if the loop has not run."""
        return self.latest if self.latest else self.tick()

    # ------------------------------------------------------------------ reports
    def mission_summary(self) -> dict:
        if not self.history:
            return {'samples': 0}
        health = [s['digital_twin']['health_score'] for s in self.history]
        faults: dict[str, int] = {}
        for snapshot in self.history:
            fault = snapshot['prediction']['predicted_fault']
            faults[fault] = faults.get(fault, 0) + 1
        first, last = self.history[0], self.history[-1]
        subsystems = last['digital_twin']['subsystems']
        return {
            'samples': len(self.history),
            'profile': last['mission_profile'],
            'ingestion': self.ingestion_mode,
            'engine_hours_start': first['telemetry'].get('engine_hours'),
            'engine_hours_end': last['telemetry'].get('engine_hours'),
            'health_start': first['digital_twin']['health_score'],
            'health_end': last['digital_twin']['health_score'],
            'health_min': round(float(np.min(health)), 1),
            'health_mean': round(float(np.mean(health)), 1),
            'fault_sample_counts': faults,
            'anomaly_sample_fraction': round(
                float(np.mean([1.0 if s['prediction']['anomaly'] else 0.0 for s in self.history])), 3),
            'final_subsystem_health': {k: v['health'] for k, v in subsystems.items()},
            'final_rul': {'hours': last['rul']['hours'],
                          'confidence_interval_hours': last['rul']['confidence_interval_hours'],
                          'limiting_subsystem': last['rul']['limiting_subsystem']},
            'advisory': last['advisory'],
            'alarms': list(self.alarm_log),
            'peak_readings': {
                'cht_max': round(max(s['telemetry']['cht'] for s in self.history), 1),
                'egt_max': round(max(s['telemetry']['egt'] for s in self.history), 1),
                'oil_temperature_max': round(max(s['telemetry']['oil_temperature'] for s in self.history), 1),
                'oil_pressure_min': round(min(s['telemetry']['oil_pressure'] for s in self.history), 1),
                'vibration_max': round(max(s['telemetry']['vibration'] for s in self.history), 3),
            },
        }

    def mission_report(self, mission_id: int) -> dict | None:
        mission = self.db.mission(mission_id)
        if mission is None:
            return None
        if mission.get('summary'):
            return {'mission': mission, 'summary': mission['summary'], 'source': 'stored'}
        snapshots = self.db.snapshots(mission_id, 0, settings.history_limit)
        if not snapshots:
            return {'mission': mission, 'summary': {'samples': 0}, 'source': 'empty'}
        health = [s['digital_twin']['health_score'] for s in snapshots]
        return {'mission': mission, 'source': 'derived', 'summary': {
            'samples': len(snapshots),
            'health_min': round(float(np.min(health)), 1),
            'health_mean': round(float(np.mean(health)), 1),
            'health_end': snapshots[-1]['digital_twin']['health_score'],
            'advisory': snapshots[-1]['advisory'],
            'final_rul': snapshots[-1]['rul'],
        }}

    def close(self) -> None:
        self.sim.running = False
        self.source.close()
