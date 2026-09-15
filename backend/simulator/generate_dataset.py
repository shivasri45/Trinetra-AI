"""Build the labelled training set for the diagnostic models.

Each session warms the engine up healthy, lets the Kalman filters converge,
then injects one fault and records its full development from incipient to
expressed. Sessions are run across every mission profile - including the rapid
throttle transition profile - so the classifier sees healthy transients and
learns not to treat them as faults.

Only features from ``backend.preprocessing.pipeline`` are written, plus the
label columns. The label is never fed back into the feature builder.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from backend.config.settings import FAULTS, MISSION_PROFILES, settings
from backend.digital_twin.synchroniser import TwinSynchroniser
from backend.preprocessing.pipeline import FEATURE_NAMES
from backend.simulator.engine import EngineSimulator

WARMUP_TICKS = 150      # thermal soak plus filter convergence
SETTLE_TICKS = 40       # discarded, filters still converging
FAULT_TICKS = 230       # covers incipient through fully developed
NORMAL_TAIL = 60        # healthy samples recorded before injection


def run_session(mission: str, fault: str, seed: int, cylinder: int = 2,
                session: str | None = None, jitter: bool = False,
                fault_ticks: int = FAULT_TICKS) -> list[dict]:
    sim = EngineSimulator(seed)
    sim.affected_cylinder = cylinder
    sim.configure_mission(mission)
    if jitter:
        # Spread the healthy envelope so the novelty detector learns the whole
        # manifold rather than five discrete operating points.
        rng = np.random.default_rng(seed)
        sim.throttle_command = float(np.clip(sim.throttle_command + rng.normal(0, 0.10), 0.40, 1.0))
        sim.ambient_temperature += float(rng.normal(0, 9.0))
        sim.ambient_pressure = float(np.clip(sim.ambient_pressure + rng.normal(0, 9.0), 58.0, 103.0))
        sim.airspeed_ratio = float(np.clip(sim.airspeed_ratio + rng.normal(0, 0.09), 0.65, 1.25))
    sync = TwinSynchroniser()
    rows: list[dict] = []

    for _ in range(WARMUP_TICKS):
        frame = sim.step()
        sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)

    for tick in range(NORMAL_TAIL + fault_ticks):
        if tick == NORMAL_TAIL:
            sim.inject_fault(fault)
        frame = sim.step()
        result = sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)
        if tick < SETTLE_TICKS:
            continue
        label = 'normal' if tick < NORMAL_TAIL else fault
        row = {name: round(result.features.get(name, 0.0), 6) for name in FEATURE_NAMES}
        row['fault'] = label
        row['severity'] = frame.truth['severity'] if tick >= NORMAL_TAIL else 0.0
        row['mission'] = mission
        row['health'] = result.twin['health_score']
        row['session'] = session or f'{mission}|{fault}|c{cylinder}|s{seed}'
        rows.append(row)
    return rows


def generate(path: Path | None = None, verbose: bool = True) -> Path:
    target = Path(path or settings.dataset_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seed = 1000
    for mission in MISSION_PROFILES:
        for fault in FAULTS:
            # Two independent builds per case, and two affected cylinders for the
            # cylinder-specific faults, so sessions can be held out cleanly.
            cylinders = (2, 0) if fault in ('misfire', 'injector_abnormality') else (2, 1)
            for repeat, cylinder in enumerate(cylinders):
                seed += 1
                # Deliberately NOT jittered. Widening the fault sessions' operating
                # envelope with the same spread used for healthy runs was measured
                # and made things worse: overall accuracy 0.981 -> 0.950 and
                # sensor_drift F1 0.909 -> 0.000, i.e. the class stopped being
                # predicted at all. Sensor drift is a CHT bias, and an ambient
                # temperature spread of sigma 9 K is large enough to swamp it.
                # Fault coverage across the envelope is still worth having, but it
                # needs a spread that does not mask the subtlest fault - see the
                # note in docs/model_card.md.
                rows += run_session(mission, fault, seed, cylinder,
                                    session=f'{mission}|{fault}|r{repeat}')
        if verbose:
            print(f'  {mission}: {len(rows)} cumulative rows')

    # Extra healthy sessions with a randomised envelope. Healthy behaviour has to
    # be covered far more densely than any single fault, because the novelty
    # detector's decision boundary is drawn entirely from it.
    for repeat in range(6):
        for mission in MISSION_PROFILES:
            seed += 1
            rows += run_session(mission, 'normal', seed, cylinder=repeat % 4,
                                session=f'healthy|{mission}|j{repeat}', jitter=True,
                                fault_ticks=180)
    if verbose:
        print(f'  healthy envelope sweep: {len(rows)} cumulative rows')

    fieldnames = list(FEATURE_NAMES) + ['fault', 'severity', 'mission', 'health', 'session']
    with open(target, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if verbose:
        print(f'wrote {len(rows)} rows and {len(FEATURE_NAMES)} features to {target}')
    return target


if __name__ == '__main__':
    generate()
