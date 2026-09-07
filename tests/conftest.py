"""Shared fixtures.

The database path is redirected before any backend module is imported, so the
suite never touches the working ``data/trinetra.db``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix='trinetra-tests-'))
os.environ['TRINETRA_DB'] = str(_TMP / 'test.db')
os.environ.setdefault('TRINETRA_API_KEY', 'test-key')
os.environ.setdefault('TRINETRA_HMAC_KEY', 'test-hmac')
os.environ.setdefault('TRINETRA_AUTH', '1')

import pytest  # noqa: E402

from backend.digital_twin.synchroniser import TwinSynchroniser  # noqa: E402
from backend.ml.diagnostics import Diagnostics  # noqa: E402
from backend.simulator.engine import EngineSimulator  # noqa: E402

API_KEY = os.environ['TRINETRA_API_KEY']
AUTH = {'X-API-Key': API_KEY}


@pytest.fixture
def tmp_db_path(tmp_path) -> Path:
    return tmp_path / 'mission.db'


def run_pipeline(fault: str = 'normal', mission: str = 'normal_cruise', warmup: int = 120,
                 after: int = 140, seed: int = 4242, with_diagnostics: bool = False):
    """Drive simulator -> twin -> features (-> diagnostics) and return the last state."""
    sim = EngineSimulator(seed)
    sim.configure_mission(mission)
    sync = TwinSynchroniser()
    diagnostics = Diagnostics() if with_diagnostics else None
    result = prediction = frame = None

    for _ in range(warmup):
        frame = sim.step()
        result = sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)
        if diagnostics:
            prediction = diagnostics.analyse(result.features, result.twin)

    if fault != 'normal':
        sim.inject_fault(fault)

    for _ in range(after):
        frame = sim.step()
        result = sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)
        if diagnostics:
            prediction = diagnostics.analyse(result.features, result.twin)

    return frame, result, prediction


@pytest.fixture(scope='session')
def diagnostics_available() -> bool:
    return Diagnostics().backend == 'trained'
