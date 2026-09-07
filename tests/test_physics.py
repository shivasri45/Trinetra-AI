"""The thermodynamic model must obey the relationships it claims to implement."""
from __future__ import annotations

import math

import pytest

from backend.config.settings import settings
from backend.simulator import physics
from backend.simulator.physics import Commands, EnginePhysics, Environment

SPEC = settings.engine


@pytest.fixture
def engine() -> EnginePhysics:
    return EnginePhysics(SPEC)


def test_air_density_matches_ideal_gas():
    # ISA sea level: 101.325 kPa, 15 C -> 1.225 kg/m3
    assert physics.air_density(101.325, 15.0) == pytest.approx(1.225, rel=1e-3)
    assert physics.density_ratio(101.325, 15.0) == pytest.approx(1.0, rel=1e-3)


def test_density_falls_with_altitude_and_heat():
    assert physics.air_density(68.0, -8.0) < physics.air_density(101.325, 15.0)
    assert physics.air_density(101.325, 45.0) < physics.air_density(101.325, 15.0)


def test_pressure_altitude_is_inverse_of_isa():
    assert physics.pressure_altitude_m(101.325) == pytest.approx(0.0, abs=1.0)
    assert physics.pressure_altitude_m(54.0) == pytest.approx(5000.0, abs=250.0)


def test_otto_efficiency_matches_closed_form():
    expected = 1.0 - SPEC.compression_ratio ** (1.0 - physics.GAMMA)
    assert physics.otto_efficiency(SPEC.compression_ratio) == pytest.approx(expected)
    # Higher compression is more efficient, and the value stays physical.
    assert physics.otto_efficiency(12.0) > physics.otto_efficiency(8.0)
    assert 0.0 < physics.otto_efficiency(SPEC.compression_ratio) < 1.0


def test_volumetric_efficiency_peaks_at_tuned_speed():
    peak = physics.volumetric_efficiency(physics.VE_PEAK_RPM)
    assert peak > physics.volumetric_efficiency(1500.0)
    assert peak > physics.volumetric_efficiency(5800.0)
    assert physics.volumetric_efficiency(4000.0, restriction=0.6) < physics.volumetric_efficiency(4000.0)


def test_four_stroke_mass_flow_scales_with_speed_and_density():
    base = physics.air_mass_flow(4000.0, 95.0, 30.0, SPEC)
    assert physics.air_mass_flow(2000.0, 95.0, 30.0, SPEC) < base
    assert physics.air_mass_flow(4000.0, 70.0, 30.0, SPEC) < base
    # A four-stroke inducts once per two revolutions.
    manual = (physics.air_density(95.0, 30.0) * physics.volumetric_efficiency(4000.0)
              * SPEC.displacement_m3 * (4000.0 / 60.0) / 2.0)
    assert base == pytest.approx(manual, rel=1e-9)


def test_combustion_efficiency_limited_rich_and_lean():
    stoich = physics.combustion_efficiency(SPEC.stoich_afr, SPEC)
    assert stoich == pytest.approx(1.0)
    assert physics.combustion_efficiency(11.0, SPEC) < stoich       # oxygen limited
    assert physics.combustion_efficiency(21.0, SPEC) < stoich       # past the lean limit


def test_rated_power_is_reproduced_at_sea_level(engine):
    point = engine.operating_point(Commands(throttle=1.0), Environment(15.0, 101.325, 1.0))
    assert point.rpm == pytest.approx(SPEC.max_rpm, rel=0.05)
    assert point.power_kw == pytest.approx(SPEC.rated_power_kw, rel=0.12)
    # Brake specific fuel consumption in the range real piston engines achieve.
    assert 0.20 < point.bsfc_kg_kwh < 0.34


def test_power_lapses_with_altitude(engine):
    sea = engine.operating_point(Commands(throttle=1.0), Environment(15.0, 101.325, 1.0))
    high = engine.operating_point(Commands(throttle=1.0), Environment(-8.0, 68.0, 1.0))
    assert high.power_kw < sea.power_kw
    assert high.rpm < sea.rpm
    assert high.density_ratio < sea.density_ratio


def test_throttle_monotonically_increases_speed_and_power(engine):
    env = Environment(18.0, 95.0, 1.0)
    points = [engine.operating_point(Commands(throttle=t), env) for t in (0.4, 0.6, 0.8, 1.0)]
    assert [p.rpm for p in points] == sorted(p.rpm for p in points)
    assert [p.power_kw for p in points] == sorted(p.power_kw for p in points)


def test_equilibrium_balances_available_and_required_power(engine):
    cmd, env = Commands(throttle=0.7), Environment(20.0, 97.0, 1.0)
    rpm = engine.equilibrium_rpm(cmd, env)
    available = engine.available_power(rpm, cmd, env)[0]
    required = engine.required_power(rpm, cmd)
    assert available == pytest.approx(required, rel=1e-3)


def test_egt_peaks_slightly_lean_of_stoichiometric(engine):
    """The classic mixture sweep: EGT rises to a peak then falls when too lean."""
    env = Environment(18.0, 95.0, 1.0)
    sweep = [(bias, engine.steady_state(Commands(throttle=0.68, lean_bias=bias), env))
             for bias in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    egts = [state['egt'] for _, state in sweep]
    afrs = [state['air_fuel_ratio'] for _, state in sweep]
    peak = egts.index(max(egts))

    assert 0 < peak < len(egts) - 1, 'EGT should peak inside the sweep, not at an end'
    assert 14.5 < afrs[peak] < 17.5, 'peak EGT should sit slightly lean of stoichiometric'
    assert egts[0] < egts[peak], 'rich mixture must run cooler than peak'
    assert egts[-1] < egts[peak], 'over-lean mixture must fall away from peak'


def test_enrichment_cools_the_cylinder_head(engine):
    """Power enrichment is a cooling measure, so richer must mean a cooler head."""
    env = Environment(18.0, 95.0, 1.0)
    rich = engine.steady_state(Commands(throttle=0.68, lean_bias=-2.0), env)
    lean = engine.steady_state(Commands(throttle=0.68, lean_bias=+2.0), env)
    assert rich['cht'] <= lean['cht'] or rich['fuel_flow'] > lean['fuel_flow']


def test_losing_cooling_airflow_raises_temperatures(engine):
    cmd = Commands(throttle=0.7)
    normal = engine.steady_state(cmd, Environment(20.0, 97.0, 1.0))
    blocked = engine.steady_state(cmd, Environment(20.0, 97.0, 0.45))
    assert blocked['cht'] > normal['cht']
    assert blocked['oil_temperature'] > normal['oil_temperature']


def test_oil_pressure_falls_with_temperature_and_rises_with_speed(engine):
    assert engine.oil_pressure(4000.0, 120.0) < engine.oil_pressure(4000.0, 80.0)
    assert engine.oil_pressure(2000.0, 90.0) < engine.oil_pressure(4500.0, 90.0)
    assert engine.oil_pressure(6000.0, 60.0) <= physics.OIL_RELIEF_PSI
    assert engine.oil_pressure(4000.0, 90.0, spec_leak=25.0) < engine.oil_pressure(4000.0, 90.0)


def test_oil_viscosity_thins_when_hot():
    assert physics.EnginePhysics.oil_viscosity_ratio(120.0) < 1.0
    assert physics.EnginePhysics.oil_viscosity_ratio(60.0) > 1.0
    assert physics.EnginePhysics.oil_viscosity_ratio(90.0) == pytest.approx(1.0)


def test_intake_restriction_reduces_breathing_and_power(engine):
    env = Environment(18.0, 95.0, 1.0)
    clean = engine.operating_point(Commands(throttle=0.7), env)
    coked = engine.operating_point(Commands(throttle=0.7, intake_restriction=0.55), env)
    assert coked.volumetric_efficiency < clean.volumetric_efficiency
    assert coked.power_kw < clean.power_kw
    assert coked.fuel_flow_lph < clean.fuel_flow_lph


def test_uncompensated_metering_goes_rich_at_altitude():
    sigma = physics.density_ratio(68.0, -8.0)
    uncompensated = physics.effective_afr(0.9, sigma, compensation=0.0)
    compensated = physics.effective_afr(0.9, sigma, compensation=1.0)
    assert uncompensated < compensated


def test_cylinder_temperatures_localise_faults(engine):
    env = Environment(18.0, 95.0, 1.0)
    state = engine.steady_state(Commands(throttle=0.68), env)
    n = SPEC.cylinders
    args = (state['air_flow_kg_s'], state['fuel_flow_kg_s'], state['air_fuel_ratio'], env, 0.6)

    egt_even, _ = engine.cylinder_temperatures(*args, [0.0] * n, [1.0] * n, [1.0] * n)
    assert max(egt_even) - min(egt_even) < 1.0

    dead = [1.0] * n
    dead[2] = 0.0
    egt_dead, cht_dead = engine.cylinder_temperatures(*args, [0.0] * n, dead, [1.0] * n)
    assert egt_dead[2] < min(egt_dead[0], egt_dead[1], egt_dead[3]) - 200.0
    assert cht_dead[2] < cht_dead[0]

    lean = [0.0] * n
    lean[2] = 2.5
    egt_lean, _ = engine.cylinder_temperatures(*args, lean, [1.0] * n, [1.0] * n)
    assert egt_lean[2] > max(egt_lean[0], egt_lean[1], egt_lean[3])


def test_instrument_readings_land_in_realistic_ranges(engine):
    """Guards against a refactor silently producing unphysical gauge values."""
    cases = {
        'cruise': (Commands(throttle=0.68), Environment(18.0, 95.0, 1.00)),
        'full power': (Commands(throttle=1.00), Environment(15.0, 101.325, 1.00)),
        'high altitude': (Commands(throttle=0.92), Environment(-8.0, 68.0, 1.05)),
        'hot and slow': (Commands(throttle=0.70), Environment(44.0, 99.0, 0.85)),
    }
    for name, (cmd, env) in cases.items():
        state = engine.steady_state(cmd, env)
        assert 2000.0 < state['rpm'] <= SPEC.max_rpm * 1.02, name
        assert 90.0 < state['cht'] < 300.0, f'{name}: CHT {state["cht"]}'
        assert 450.0 < state['egt'] < 900.0, f'{name}: EGT {state["egt"]}'
        assert 40.0 < state['oil_temperature'] < 150.0, f'{name}: oil {state["oil_temperature"]}'
        assert 10.0 < state['oil_pressure'] <= physics.OIL_RELIEF_PSI, name
        assert 4.0 < state['fuel_flow'] < 40.0, name
        assert 8.0 < state['air_fuel_ratio'] < 20.0, name
        assert 5.0 < state['injection_timing'] < 35.0, name
        assert not math.isnan(state['power_kw'])
