"""Simulator behaviour: determinism, thermal inertia, wear and fault signatures."""
from __future__ import annotations

import pytest

from backend.config.settings import FAULTS, MISSION_PROFILES, settings
from backend.simulator.engine import EngineSimulator


from backend.simulator.engine import FAULT_RAMP


def settle(fault: str = 'normal', mission: str = 'normal_cruise', warmup: int = 120,
           after: int = 150, seed: int = 7):
    sim = EngineSimulator(seed)
    sim.configure_mission(mission)
    frame = None
    for _ in range(warmup):
        frame = sim.step()
    baseline = frame
    if fault != 'normal':
        sim.inject_fault(fault)
    for _ in range(after):
        frame = sim.step()
    return baseline, frame, sim


def paired(fault: str, mission: str = 'normal_cruise', warmup: int = 120,
           after: int | None = None, seed: int = 7):
    """Run an identical healthy control alongside the faulted engine.

    Comparing against a control at the same tick isolates the fault's effect from
    residual warm-up drift, which matters for slow channels such as oil
    temperature whose time constant is over a minute.
    """
    if after is None:
        after = int(FAULT_RAMP.get(fault, 30.0)) + 60
    control = EngineSimulator(seed)
    faulted = EngineSimulator(seed)
    for sim in (control, faulted):
        sim.configure_mission(mission)
    for _ in range(warmup):
        control.step()
        faulted.step()
    faulted.inject_fault(fault)
    control_frame = faulted_frame = None
    for _ in range(after):
        control_frame = control.step()
        faulted_frame = faulted.step()
    return control_frame, faulted_frame, faulted


def test_simulator_is_deterministic_for_a_seed():
    a, b = EngineSimulator(11), EngineSimulator(11)
    for _ in range(30):
        left, right = a.step(), b.step()
        assert left.telemetry.rpm == right.telemetry.rpm
        assert left.telemetry.cht == right.telemetry.cht
        assert left.vibration_features['overall_rms'] == right.vibration_features['overall_rms']


def test_different_seeds_give_different_build_tolerances():
    a, b = EngineSimulator(1), EngineSimulator(2)
    assert a.cyl_trim != b.cyl_trim


def test_reset_restores_initial_state():
    sim = EngineSimulator(5)
    for _ in range(40):
        sim.step()
    sim.inject_fault('overheating')
    sim.step()
    sim.reset()
    assert sim.t == 0
    assert sim.fault == 'normal'
    assert sim.engine_hours == 0.0
    assert sim.state is None
    assert all(value == 0.0 for value in sim.wear.values())


def test_cylinder_head_lags_the_steady_state_target():
    """Thermal mass means CHT must not jump instantly on a throttle change."""
    sim = EngineSimulator(3)
    sim.configure_mission('normal_cruise')
    for _ in range(200):
        sim.step()
    before = sim.step().telemetry.cht
    sim.throttle_command = 1.0
    after = sim.step().telemetry.cht
    assert abs(after - before) < 12.0, 'CHT changed too fast for a thermal mass'


def test_speed_responds_faster_than_head_temperature():
    sim = EngineSimulator(3)
    sim.configure_mission('normal_cruise')
    for _ in range(200):
        sim.step()
    base = sim.step().telemetry
    sim.throttle_command = 1.0
    for _ in range(6):
        frame = sim.step()
    rpm_fraction = (frame.telemetry.rpm - base.rpm) / max(1.0, base.rpm)
    cht_fraction = (frame.telemetry.cht - base.cht) / max(1.0, base.cht)
    assert rpm_fraction > cht_fraction


@pytest.mark.parametrize('mission', MISSION_PROFILES)
def test_every_mission_profile_runs_and_stays_physical(mission):
    _, frame, _ = settle(mission=mission, warmup=150, after=60)
    telemetry = frame.telemetry
    assert 2000 < telemetry.rpm <= settings.engine.max_rpm * 1.05
    assert 60 < telemetry.cht < 320
    assert 400 < telemetry.egt < 950
    assert telemetry.oil_pressure > 5
    assert telemetry.fuel_flow > 2
    assert telemetry.engine_hours > 0


@pytest.mark.parametrize('fault', [f for f in FAULTS if f != 'normal'])
def test_every_fault_perturbs_the_engine(fault):
    control, frame, sim = paired(fault)
    assert sim.severity == pytest.approx(1.0, abs=1e-6)
    telemetry, base = frame.telemetry, control.telemetry
    signals = (
        abs(telemetry.rpm - base.rpm) / 40.0,
        abs(telemetry.cht - base.cht) / 4.0,
        abs(telemetry.egt - base.egt) / 12.0,
        abs(telemetry.oil_pressure - base.oil_pressure) / 1.5,
        abs(telemetry.oil_temperature - base.oil_temperature) / 3.0,
        abs(telemetry.fuel_flow - base.fuel_flow) / 0.8,
        abs(telemetry.vibration - base.vibration) / 0.05,
        abs(telemetry.battery_voltage - base.battery_voltage) / 0.2,
        abs(telemetry.egt_spread - base.egt_spread) / 25.0,
    )
    assert max(signals) > 1.0, f'{fault} produced no measurable change'


def test_misfire_puts_energy_at_the_half_crank_order():
    baseline, frame, _ = settle(fault='misfire')
    assert frame.vibration_features['misfire_index'] > 5 * baseline.vibration_features['misfire_index']
    assert frame.vibration_features['misfire_index'] > 0.5


def test_misfire_cools_one_cylinder_and_loses_power():
    baseline, frame, _ = settle(fault='misfire')
    egts = frame.cylinders['egt']
    coldest = min(egts)
    assert coldest < min(sorted(egts)[1:]) - 200
    assert frame.telemetry.rpm < baseline.telemetry.rpm - 200
    assert frame.telemetry.egt_spread > 200


def test_lean_injector_heats_one_cylinder():
    _, frame, _ = settle(fault='injector_abnormality')
    egts = frame.cylinders['egt']
    assert max(egts) - min(egts) > 40
    assert frame.telemetry.hottest_cylinder == 3  # the affected cylinder


def test_imbalance_appears_at_the_first_order():
    baseline, frame, _ = settle(fault='excessive_vibration')
    assert frame.vibration_features['imbalance_index'] > 4 * baseline.vibration_features['imbalance_index']


def test_lubrication_fault_raises_broadband_energy_and_drops_pressure():
    baseline, frame, _ = settle(fault='lubrication_issue')
    assert frame.telemetry.oil_pressure < baseline.telemetry.oil_pressure - 15
    assert frame.telemetry.oil_temperature > baseline.telemetry.oil_temperature
    assert frame.vibration_features['broadband_rms'] > 3 * baseline.vibration_features['broadband_rms']


def test_overheating_moves_the_whole_thermal_chain():
    """A real thermal event must move head and oil temperature together."""
    baseline, frame, _ = settle(fault='overheating')
    assert frame.telemetry.cht > baseline.telemetry.cht + 60
    assert frame.telemetry.oil_temperature > baseline.telemetry.oil_temperature + 30


def test_sensor_drift_is_applied_only_on_the_sensor_side():
    """The distinguishing property: CHT reads high while its coupled channels do not."""
    control, frame, _ = paired('sensor_drift')
    assert frame.telemetry.cht > control.telemetry.cht + 30
    assert frame.telemetry.oil_temperature == pytest.approx(control.telemetry.oil_temperature, abs=1.0)
    assert frame.telemetry.egt == pytest.approx(control.telemetry.egt, abs=15.0)
    assert frame.telemetry.oil_pressure == pytest.approx(control.telemetry.oil_pressure, abs=1.5)


def test_overheating_and_sensor_drift_are_physically_distinguishable():
    """Both raise CHT; only the real thermal event moves the rest of the chain."""
    control, drift, _ = paired('sensor_drift')
    _, overheat, _ = paired('overheating')
    drift_oil = drift.telemetry.oil_temperature - control.telemetry.oil_temperature
    overheat_oil = overheat.telemetry.oil_temperature - control.telemetry.oil_temperature
    assert abs(drift_oil) < 2.0
    assert overheat_oil > 20.0


def test_coking_reduces_breathing_speed_and_fuel_flow():
    baseline, frame, _ = settle(fault='coking_degradation', after=260)
    assert frame.telemetry.rpm < baseline.telemetry.rpm - 200
    assert frame.telemetry.fuel_flow < baseline.telemetry.fuel_flow - 1.5


def test_alternator_failure_discharges_the_battery():
    _, frame, _ = settle(fault='alternator_failure')
    assert frame.telemetry.alternator_output < 4.0
    assert frame.telemetry.battery_voltage < 12.5


def test_wear_accumulates_and_hot_operation_accelerates_it():
    _, cool, _ = settle(mission='long_endurance', warmup=60, after=140)
    _, hot, _ = settle(mission='hot_weather', warmup=60, after=140)
    assert cool.wear['thermal'] > 0.0
    assert hot.wear['thermal'] > cool.wear['thermal'], 'Arrhenius acceleration not observed'


def test_engine_hours_advance_monotonically():
    sim = EngineSimulator(9)
    hours = [sim.step().telemetry.engine_hours for _ in range(20)]
    assert hours == sorted(hours)
    assert hours[-1] > hours[0]


def test_injecting_an_unknown_fault_is_rejected():
    with pytest.raises(ValueError):
        EngineSimulator(1).inject_fault('not_a_fault')


def test_severity_ramps_from_zero_to_one():
    sim = EngineSimulator(2)
    sim.configure_mission('normal_cruise')
    for _ in range(30):
        sim.step()
    assert sim.severity == 0.0
    sim.inject_fault('overheating')
    sim.step()
    early = sim.severity
    assert 0.0 < early < 1.0
    for _ in range(120):
        sim.step()
    assert sim.severity == pytest.approx(1.0)


def test_vibration_waveform_has_expected_shape():
    frame = EngineSimulator(4).step()
    assert frame.waveform.shape == (settings.vib_block_samples,)
    features = frame.vibration_features
    assert features['overall_rms'] > 0
    assert features['order_2p0'] > features['order_0p5'], 'firing order should dominate when healthy'
    assert features['rotational_hz'] > 0
    assert features['firing_hz'] == pytest.approx(2 * features['rotational_hz'], rel=0.01)
