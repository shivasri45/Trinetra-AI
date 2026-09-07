"""Digital twin: state estimation, health indices and sensor validation."""
from __future__ import annotations

import numpy as np
import pytest

from backend.config.settings import MISSION_PROFILES, SUBSYSTEMS, settings
from backend.digital_twin.estimator import ChannelFilter, StateEstimator
from backend.digital_twin.synchroniser import TwinSynchroniser
from backend.simulator.engine import EngineSimulator
from conftest import run_pipeline


# --------------------------------------------------------------------- filter
def test_filter_tracks_a_constant_input():
    filt = ChannelFilter('test', tau=5.0, q=0.01, r=0.04, dt=1.0)
    for _ in range(200):
        estimate = filt.step(100.0, 100.0)
    assert estimate.estimate == pytest.approx(100.0, abs=0.5)
    assert abs(estimate.normalised) < 1.0


def test_filter_innovations_are_whitened_on_a_matched_model():
    """A correctly tuned filter should produce mean NIS on the order of one."""
    rng = np.random.default_rng(0)
    filt = ChannelFilter('test', tau=8.0, q=0.25, r=1.0, dt=1.0)
    truth, nis = 50.0, []
    for step in range(600):
        target = 50.0 + 10.0 * np.sin(step / 40.0)
        truth += (target - truth) * (1.0 / 8.0) + rng.normal(0, 0.5)
        estimate = filt.step(target, truth + rng.normal(0, 1.0))
        if step > 100:
            nis.append(estimate.nis)
    assert 0.2 < float(np.mean(nis)) < 4.0


def test_filter_absorbs_small_offsets_into_the_learned_baseline():
    filt = ChannelFilter('test', tau=4.0, q=0.02, r=0.25, dt=1.0)
    for _ in range(1200):
        estimate = filt.step(100.0, 100.4)
    assert abs(estimate.baseline_bias) > 0.05
    assert abs(estimate.normalised) < abs(estimate.innovation / estimate.innovation_sigma) + 1e-9


def test_estimator_covers_every_declared_channel():
    estimator = StateEstimator()
    sim = EngineSimulator(3)
    frame = sim.step()
    from backend.digital_twin.twin import DigitalTwin
    expected, cyl_expected = DigitalTwin().expected_state(frame.telemetry.dict())
    out = estimator.step(frame.telemetry.dict(), expected, frame.cylinders, cyl_expected)
    for channel in ('rpm', 'cht', 'egt', 'oil_pressure', 'oil_temperature', 'fuel_flow',
                    'air_fuel_ratio', 'manifold_pressure', 'vibration',
                    'battery_voltage', 'alternator_output', 'injection_timing'):
        assert channel in out
    assert 'egt_cyl1' in out and f'cht_cyl{settings.engine.cylinders}' in out


# ----------------------------------------------------------------------- twin
def test_twin_has_no_access_to_the_injected_fault():
    """The expectation is a function of the telemetry frame alone."""
    from backend.digital_twin.twin import DigitalTwin
    sim = EngineSimulator(3)
    for _ in range(60):
        frame = sim.step()
    telemetry = frame.telemetry.dict()
    clean = DigitalTwin().expected_state(telemetry)[0]

    sim.inject_fault('overheating')
    for _ in range(40):
        faulted = sim.step()
    # Feed the faulted frame's *commands* but with identical measured inputs:
    # the expectation must depend only on throttle and atmosphere.
    probe = dict(faulted.telemetry.dict())
    probe.update({k: telemetry[k] for k in ('throttle', 'ambient_temperature',
                                            'ambient_pressure', 'airspeed_ratio')})
    again = DigitalTwin().expected_state(probe)[0]
    assert again['cht'] == pytest.approx(clean['cht'], rel=1e-9)


@pytest.mark.parametrize('mission', MISSION_PROFILES)
def test_healthy_engine_reports_healthy_on_every_mission(mission):
    _, result, _ = run_pipeline(mission=mission, warmup=180, after=120)
    twin = result.twin
    assert twin['health_score'] > settings.warning_health, twin['subsystems']
    assert twin['operating_state'] == 'NORMAL'
    assert twin['mean_nis'] < settings.nis_warning
    # A single channel may momentarily exceed its band during an aggressive
    # throttle step. That is why alarms are debounced rather than raised on one
    # sample; the end-to-end false-alarm contract is asserted in test_diagnostics.
    tolerance = 1 if mission == 'rapid_throttle_transition' else 0
    assert len(twin['abnormality_indicators']) <= tolerance, twin['abnormality_indicators']


def test_kalman_filter_suppresses_transient_false_alarms():
    """The headline claim: naive expectation comparison fails in transients.

    A steady-state expectation is compared against the measurement directly, and
    against the filter's transient-aware innovation, during the rapid throttle
    profile. The naive residual must be dramatically worse.
    """
    sim = EngineSimulator(31)
    sim.configure_mission('rapid_throttle_transition')
    sync = TwinSynchroniser()
    naive, filtered = [], []
    for step in range(360):
        frame = sim.step()
        telemetry = frame.telemetry.dict()
        result = sync.process(telemetry, frame.cylinders, frame.vibration_features)
        if step <= 160:
            continue
        expected = result.twin['expected']
        naive.append(max(abs(telemetry['cht'] - expected['cht']) / 2.0,
                         abs(telemetry['rpm'] - expected['rpm']) / 10.0))
        filtered.append(max(abs(result.twin['normalised_residuals'][c])
                            for c in ('cht', 'rpm')))
    assert max(naive) > 25.0, 'the naive comparison should be badly wrong in transients'
    assert max(filtered) < 6.0, 'the filter should stay near the noise floor'
    assert max(naive) > 5 * max(filtered)


def test_subsystem_indices_are_all_present_and_bounded():
    _, result, _ = run_pipeline(warmup=150, after=60)
    subsystems = result.twin['subsystems']
    assert set(subsystems) == set(SUBSYSTEMS)
    for name, value in subsystems.items():
        assert 0.0 <= value['health'] <= 100.0, name
        assert value['acute_penalty'] >= 0.0
        assert value['chronic_penalty'] >= 0.0


@pytest.mark.parametrize('fault,subsystem', [
    ('overheating', 'thermal'),
    ('lubrication_issue', 'lubrication'),
    ('misfire', 'combustion'),
    ('excessive_vibration', 'mechanical'),
    ('alternator_failure', 'electrical'),
])
def test_fault_degrades_the_responsible_subsystem_most(fault, subsystem):
    _, result, _ = run_pipeline(fault=fault, warmup=150, after=170)
    subsystems = result.twin['subsystems']
    worst = min(subsystems, key=lambda key: subsystems[key]['health'])
    assert worst == subsystem, f'{fault} blamed {worst}, expected {subsystem}: ' \
                               f'{ {k: v["health"] for k, v in subsystems.items()} }'


def test_a_cold_cylinder_is_blamed_on_combustion_not_thermal():
    """Direction matters: a dead cylinder is cold, and cold is not overheating."""
    _, result, _ = run_pipeline(fault='misfire', warmup=150, after=150)
    subsystems = result.twin['subsystems']
    assert subsystems['combustion']['health'] < subsystems['thermal']['health']


def test_sensor_validation_isolates_a_drifting_probe():
    _, result, _ = run_pipeline(fault='sensor_drift', warmup=180, after=220)
    validation = result.twin['sensor_validation']
    assert 'cht' in validation['isolation']
    assert validation['isolation']['cht'] > validation['isolation'].get('oil_temperature', 0.0)


def test_sensor_validation_stays_quiet_on_a_healthy_engine():
    _, result, _ = run_pipeline(warmup=200, after=120)
    assert result.twin['sensor_validation']['plausibility'] == 'CONSISTENT'


def test_real_overheating_keeps_the_thermal_chain_consistent():
    _, result, _ = run_pipeline(fault='overheating', warmup=150, after=180)
    groups = result.twin['sensor_validation']['groups']
    members = groups['thermal_chain']['members']
    assert members['cht'] > 1.0 and members['oil_temperature'] > 1.0, members


def test_transient_biased_channel_is_excluded_from_the_nis_statistic():
    """NIS is a consistency measure and is only valid for unbiased channels.

    The nominal oil-pressure expectation assumes the steady-state oil temperature,
    which a throttle-cycling engine never reaches, so including it would swamp the
    aggregate.
    """
    from backend.digital_twin.twin import NIS_EXCLUDED
    sim = EngineSimulator(31)
    sim.configure_mission('rapid_throttle_transition')
    sync = TwinSynchroniser()
    for _ in range(300):
        frame = sim.step()
        result = sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)

    twin = result.twin
    assert 'oil_pressure' in NIS_EXCLUDED
    assert set(twin['nis_excluded_channels']) == set(NIS_EXCLUDED)
    assert twin['nis_channels'] == len(twin['normalised_residuals']) - len(NIS_EXCLUDED)

    included = [v ** 2 for k, v in twin['normalised_residuals'].items() if k not in NIS_EXCLUDED]
    assert twin['mean_nis'] == pytest.approx(float(np.mean(included)), rel=0.01)
    assert twin['mean_nis'] < settings.nis_warning


def test_conditional_oil_pressure_is_the_unbiased_channel_in_transients():
    sim = EngineSimulator(31)
    sim.configure_mission('rapid_throttle_transition')
    sync = TwinSynchroniser()
    nominal, conditional = [], []
    for step in range(320):
        frame = sim.step()
        result = sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)
        if step > 150:
            nominal.append(abs(result.twin['normalised_residuals']['oil_pressure']))
            conditional.append(abs(result.twin['normalised_residuals']['oil_pressure_cond']))
    assert float(np.mean(conditional)) < float(np.mean(nominal))
    assert float(np.mean(conditional)) < 2.0


def test_oil_pressure_follows_the_actual_not_the_target_oil_temperature():
    """Pressure must not lead temperature during a throttle change."""
    from backend.simulator.physics import EnginePhysics
    physics = EnginePhysics(settings.engine)
    sim = EngineSimulator(3)
    sim.configure_mission('normal_cruise')
    for _ in range(260):
        frame = sim.step()
    telemetry = frame.telemetry
    predicted = physics.oil_pressure(telemetry.rpm, telemetry.oil_temperature,
                                     sim.oil_leak_psi)
    assert telemetry.oil_pressure == pytest.approx(predicted, abs=2.0)


def test_twin_reports_expected_estimated_and_measured_for_each_channel():
    _, result, _ = run_pipeline(warmup=120, after=40)
    twin = result.twin
    for channel in ('rpm', 'cht', 'egt', 'oil_pressure'):
        assert channel in twin['expected']
        assert channel in twin['estimated']
        assert channel in twin['measured']
        assert channel in twin['residuals']
        assert channel in twin['normalised_residuals']
