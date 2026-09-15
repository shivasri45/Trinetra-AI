"""Physics core for the aero piston engine model.

Everything here is a first-principles (or textbook-correlation) relationship
rather than a curve fit, so the same module can be used both to drive the
synthetic engine and to form the digital twin's expected state:

* ideal-gas air density and density ratio
* throttle-plate manifold pressure
* volumetric efficiency vs. engine speed
* four-stroke induction mass flow  (m_dot = rho * VE * Vd * N / 2)
* mixture schedule and oxygen-limited combustion efficiency
* air-standard Otto cycle thermal efficiency  (eta = 1 - r^(1-gamma))
* engine/propeller power balance solved for equilibrium speed
* exhaust and cylinder-head energy balances
* temperature-dependent oil viscosity and pump pressure

Units: kPa, degrees Celsius, kg/s, kW, rpm unless stated otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from backend.config.settings import EngineSpec
from backend.simulator.vibration import expected_rms, healthy_drivers

# --- Physical constants. Standard reference values, not fitted. ---
R_AIR = 287.05          # J/(kg.K), specific gas constant for dry air
RHO_SEA_LEVEL = 1.225   # kg/m3 at 15 C, 101.325 kPa (ISA sea level)
CP_EXHAUST = 1250.0     # J/(kg.K), representative for hot combustion products
GAMMA = 1.35            # polytropic index; a compromise between cold air (1.40)
                        # and hot products (~1.30), so Otto efficiency from this
                        # is indicative rather than exact

# --- Fitted constants. NOT sourced to any document. ---
# Each of these was chosen so the model lands near the reference engine's
# published rating and plausible cruise readings. They are the model's free
# parameters, and Stage 1 of docs/deployment_roadmap.md exists to replace them
# with dynamometer measurements. Anything derived from them inherits their
# uncertainty, so do not present values that depend on these as validated.
EXHAUST_HEAT_FRACTION = 0.358
HEAD_HEAT_FRACTION = 0.22
INDUCTION_HEATING_C = 15.0
CYCLE_DEVIATION = 0.78      # real cycle vs. air-standard Otto
VE_MAX = 0.92
VE_PEAK_RPM = 4800.0
VE_SPREAD_RPM = 3000.0
# Known to be wrong for a Rotax 912, which has liquid-cooled heads: this models
# forced-convection air cooling and puts cruise CHT near 184 C against a
# published limit of 135 C. Detection is unaffected because health is measured
# against the model's own expectation, but the absolute figure is not
# representative. See the validation table in docs/model_card.md.
COOLING_COEFF = 180.0       # W/K at reference cooling mass flux
OIL_PUMP_PSI_REF = 48.0     # gauge pressure at 4000 rpm, 90 C oil
OIL_RELIEF_PSI = 78.0


def air_density(pressure_kpa: float, temperature_c: float) -> float:
    """Ideal gas law: rho = p / (R * T)."""
    return (pressure_kpa * 1000.0) / (R_AIR * (temperature_c + 273.15))


def density_ratio(pressure_kpa: float, temperature_c: float) -> float:
    return air_density(pressure_kpa, temperature_c) / RHO_SEA_LEVEL


def pressure_altitude_m(pressure_kpa: float) -> float:
    """ISA inverse barometric relation, useful for operator display."""
    ratio = max(1e-3, pressure_kpa / 101.325)
    return 44330.0 * (1.0 - ratio ** 0.190284)


def manifold_pressure(ambient_kpa: float, throttle: float) -> float:
    """Throttle plate restriction; ~16 % of ambient closed, ~96 % at WOT."""
    return ambient_kpa * (0.16 + 0.80 * max(0.0, min(1.0, throttle)))


def volumetric_efficiency(rpm: float, restriction: float = 0.0) -> float:
    """Gaussian breathing curve peaking at the tuned intake speed.

    ``restriction`` (0..1) models intake/valve fouling such as carbon coking.
    """
    ve = VE_MAX * math.exp(-(((rpm - VE_PEAK_RPM) / VE_SPREAD_RPM) ** 2))
    return max(0.05, ve * (1.0 - 0.45 * restriction))


def otto_efficiency(compression_ratio: float) -> float:
    """Air-standard Otto cycle efficiency, eta = 1 - r^(1 - gamma)."""
    return 1.0 - compression_ratio ** (1.0 - GAMMA)


def mechanical_efficiency(rpm: float, spec: EngineSpec) -> float:
    """Friction mean effective pressure grows roughly linearly with speed."""
    return max(0.55, 0.92 - 1.0e-5 * rpm)


def target_afr(throttle: float, lean_bias: float = 0.0) -> float:
    """Power-enrichment schedule: rich at WOT, near best-economy at cruise."""
    throttle = max(0.0, min(1.0, throttle))
    return max(9.0, 15.2 - 2.9 * throttle ** 1.5 + lean_bias)


def effective_afr(throttle: float, sigma: float, compensation: float, lean_bias: float = 0.0) -> float:
    """Uncompensated fuel metering goes rich as air density falls with altitude."""
    base = target_afr(throttle, lean_bias)
    return max(8.0, base * (1.0 - (1.0 - compensation) * (1.0 - sigma)))


LEAN_LIMIT_AFR = 16.8
LEAN_ROLLOFF_AFR = 3.2


def combustion_efficiency(afr: float, spec: EngineSpec) -> float:
    """Fraction of injected fuel that releases useful heat.

    Two limits apply. Rich of stoichiometric the charge is oxygen limited, so
    the burnt fraction falls as AFR/AFR_stoich. Beyond the lean misfire limit
    flame propagation deteriorates and combustion becomes partial.
    """
    rich_limited = min(1.0, afr / spec.stoich_afr)
    if afr <= LEAN_LIMIT_AFR:
        return rich_limited
    return rich_limited * math.exp(-(((afr - LEAN_LIMIT_AFR) / LEAN_ROLLOFF_AFR) ** 2))


def exhaust_heat_fraction(afr: float) -> float:
    """Share of fuel energy leaving through the exhaust.

    Lean mixtures burn more slowly, so combustion finishes later in the
    expansion stroke and a larger share of the released heat is rejected to the
    exhaust instead of being converted to work. This is the mechanism behind the
    familiar EGT rise when a single cylinder runs lean.
    """
    lateness = max(0.0, min(1.3, (afr - 14.0) / 3.0))
    return EXHAUST_HEAT_FRACTION * (1.0 + 0.34 * lateness)


def air_mass_flow(rpm: float, map_kpa: float, induction_temp_c: float, spec: EngineSpec,
                  restriction: float = 0.0) -> float:
    """Four-stroke induction flow: m_dot = rho_manifold * VE * Vd * (N / 2)."""
    rho_manifold = air_density(map_kpa, induction_temp_c)
    cycles_per_s = (rpm / 60.0) / (spec.strokes / 2.0)
    return rho_manifold * volumetric_efficiency(rpm, restriction) * spec.displacement_m3 * cycles_per_s


def brake_thermal_efficiency(rpm: float, afr: float, spec: EngineSpec,
                             combustion_quality: float = 1.0) -> float:
    return (otto_efficiency(spec.compression_ratio)
            * CYCLE_DEVIATION
            * combustion_efficiency(afr, spec)
            * mechanical_efficiency(rpm, spec)
            * combustion_quality)


@dataclass
class OperatingPoint:
    rpm: float
    map_kpa: float
    air_flow_kg_s: float
    fuel_flow_kg_s: float
    fuel_flow_lph: float
    afr: float
    power_kw: float
    bsfc_kg_kwh: float
    volumetric_efficiency: float
    brake_efficiency: float
    density_ratio: float


@dataclass
class Environment:
    ambient_temperature: float = 15.0
    ambient_pressure: float = 101.325
    airspeed_ratio: float = 1.0   # cooling-air mass flux relative to cruise


@dataclass
class Commands:
    throttle: float = 0.58
    prop_load: float = 1.0             # propeller absorption factor (fixed pitch ~ 1.0)
    mixture_compensation: float = 0.85  # FADEC density compensation, 0..1
    intake_restriction: float = 0.0     # coking / filter blockage, 0..1
    combustion_quality: float = 1.0     # 1.0 = clean burn
    lean_bias: float = 0.0              # injector metering error in AFR units


class EnginePhysics:
    """Steady-state thermodynamic solution for a given command/environment pair."""

    def __init__(self, spec: EngineSpec) -> None:
        self.spec = spec

    # ------------------------------------------------------------------ power
    def available_power(self, rpm: float, cmd: Commands, env: Environment) -> tuple[float, float, float, float]:
        """Brake power at a trial speed. Returns (kW, air kg/s, fuel kg/s, AFR)."""
        map_kpa = manifold_pressure(env.ambient_pressure, cmd.throttle)
        induction_t = env.ambient_temperature + INDUCTION_HEATING_C
        sigma = density_ratio(env.ambient_pressure, env.ambient_temperature)
        afr = effective_afr(cmd.throttle, sigma, cmd.mixture_compensation, cmd.lean_bias)
        air = air_mass_flow(rpm, map_kpa, induction_t, self.spec, cmd.intake_restriction)
        fuel = air / afr
        eta = brake_thermal_efficiency(rpm, afr, self.spec, cmd.combustion_quality)
        power_kw = fuel * self.spec.fuel_lhv_mj_kg * 1000.0 * eta
        return power_kw, air, fuel, afr

    def required_power(self, rpm: float, cmd: Commands) -> float:
        """Fixed-pitch propeller absorption, P ~ rho * n^3 * D^5 -> cubic in rpm."""
        return self.spec.rated_power_kw * cmd.prop_load * (rpm / self.spec.max_rpm) ** 3

    def equilibrium_rpm(self, cmd: Commands, env: Environment) -> float:
        """Bisect the power balance P_available(N) - P_required(N) = 0."""
        lo, hi = self.spec.idle_rpm, self.spec.max_rpm * 1.06

        def imbalance(rpm: float) -> float:
            return self.available_power(rpm, cmd, env)[0] - self.required_power(rpm, cmd)

        f_lo, f_hi = imbalance(lo), imbalance(hi)
        if f_lo <= 0.0:
            return lo
        if f_hi >= 0.0:
            return hi
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if imbalance(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ----------------------------------------------------------- temperatures
    def exhaust_temperature(self, fuel_kg_s: float, air_kg_s: float, afr: float, env: Environment,
                            load_fraction: float = 0.62) -> float:
        """Energy balance on the exhaust stream with a flame-temperature factor.

        The Gaussian in AFR reproduces the familiar peak-EGT behaviour slightly
        lean of stoichiometric, the mass-flow term captures the cooling effect
        of over-rich operation, and the load term accounts for port/wall heat
        loss dominating at low power.
        """
        if fuel_kg_s <= 0.0:
            return env.ambient_temperature
        burnt = combustion_efficiency(afr, self.spec)
        flame_factor = math.exp(-0.5 * ((afr - 15.6) / 5.5) ** 2)
        wall_loss = 0.72 + 0.28 * max(0.0, min(1.2, load_fraction))
        heat_w = (fuel_kg_s * self.spec.fuel_lhv_mj_kg * 1.0e6
                  * exhaust_heat_fraction(afr) * wall_loss * burnt * (0.55 + 0.45 * flame_factor))
        mass_flow = max(1e-6, air_kg_s + fuel_kg_s)
        rise = heat_w / (mass_flow * CP_EXHAUST)
        return env.ambient_temperature + INDUCTION_HEATING_C + rise

    def head_temperature(self, fuel_kg_s: float, afr: float, env: Environment) -> float:
        """Forced-convection head cooling: Nu ~ Re^0.8 so h ~ (rho * V)^0.8.

        Only the burnt fuel fraction heats the head; surplus fuel in a rich
        mixture evaporates in the charge, which is why power enrichment is used
        as a cylinder-head cooling measure.
        """
        burnt = combustion_efficiency(afr, self.spec)
        heat_w = fuel_kg_s * self.spec.fuel_lhv_mj_kg * 1.0e6 * HEAD_HEAT_FRACTION * burnt
        sigma = density_ratio(env.ambient_pressure, env.ambient_temperature)
        cooling_flux = max(0.05, sigma * max(0.15, env.airspeed_ratio))
        conductance = COOLING_COEFF * cooling_flux ** 0.8
        return env.ambient_temperature + heat_w / conductance

    def oil_temperature(self, cht: float, power_kw: float, rpm: float, env: Environment,
                        cooling_efficiency: float = 1.0) -> float:
        """Oil picks up bearing friction heat and soaks from the head."""
        friction_kw = max(0.0, power_kw * (1.0 / max(0.5, mechanical_efficiency(rpm, self.spec)) - 1.0))
        sigma = density_ratio(env.ambient_pressure, env.ambient_temperature)
        cooler = max(0.2, sigma * max(0.15, env.airspeed_ratio)) ** 0.8 * max(0.2, cooling_efficiency)
        return env.ambient_temperature + 0.34 * (cht - env.ambient_temperature) + 3.1 * friction_kw / cooler

    @staticmethod
    def oil_viscosity_ratio(oil_temp_c: float) -> float:
        """Arrhenius-style thinning, normalised to 90 C."""
        return math.exp(-0.031 * (oil_temp_c - 90.0))

    def oil_pressure(self, rpm: float, oil_temp_c: float, spec_leak: float = 0.0) -> float:
        """Positive-displacement pump against a viscous restriction and relief valve."""
        pump = OIL_PUMP_PSI_REF * (max(0.0, rpm) / 4000.0) ** 0.6
        viscous = self.oil_viscosity_ratio(oil_temp_c) ** 0.35
        return max(0.0, min(OIL_RELIEF_PSI, pump * viscous) - spec_leak)

    def cylinder_temperatures(self, air_kg_s: float, fuel_kg_s: float, afr: float, env: Environment,
                              load_fraction: float, lean_bias: list[float], quality: list[float],
                              cooling: list[float]) -> tuple[list[float], list[float]]:
        """Per-cylinder EGT and CHT from that cylinder's own charge and burn.

        A single-probe installation only sees the collector average; per-cylinder
        probes are what let a lean injector be told apart from a dead cylinder,
        because the first drives one EGT up and the second drives one EGT down.
        """
        n = max(1, self.spec.cylinders)
        air_i = air_kg_s / n
        t_amb = env.ambient_temperature
        bulk_rise = self.head_temperature(fuel_kg_s, afr, env) - t_amb

        egts: list[float] = []
        heats: list[float] = []
        for i in range(n):
            afr_i = max(8.0, afr + lean_bias[i])
            fuel_i = air_i / afr_i
            q = max(0.0, min(1.2, quality[i]))
            if q < 0.05:
                # Dead cylinder: the charge passes through unburnt, so the probe
                # cools toward induction temperature instead of firing.
                egts.append(t_amb + INDUCTION_HEATING_C + 95.0)
                heats.append(0.06 * fuel_i)
                continue
            egts.append(self.exhaust_temperature(fuel_i * q, air_i, afr_i, env, load_fraction))
            heats.append(fuel_i * q * combustion_efficiency(afr_i, self.spec))

        # Distribute the bulk head heat load across cylinders and let each one's
        # local cooling scale its temperature rise.
        mean_heat = sum(heats) / n or 1e-9
        chts = [t_amb + bulk_rise * (h / mean_heat) / max(0.1, cooling[i]) ** 0.8
                for i, h in enumerate(heats)]
        return egts, chts

    # -------------------------------------------------------------- accessory
    def injection_advance(self, rpm: float, load: float) -> float:
        """Ignition/injection advance schedule in crank degrees BTDC."""
        return 8.0 + 22.0 * (rpm / self.spec.max_rpm) - 6.0 * max(0.0, min(1.0, load))

    def alternator_output(self, rpm: float, field_health: float = 1.0) -> float:
        """Amps, saturating once the alternator is above its cut-in speed."""
        return max(0.0, 26.0 * math.tanh(max(0.0, rpm - 900.0) / 2200.0) * field_health)

    # ------------------------------------------------------------------ solve
    def operating_point(self, cmd: Commands, env: Environment) -> OperatingPoint:
        rpm = self.equilibrium_rpm(cmd, env)
        power, air, fuel, afr = self.available_power(rpm, cmd, env)
        lph = fuel * 3600.0 / self.spec.fuel_density_kg_l
        bsfc = (fuel * 3600.0 / power) if power > 0.5 else 0.0
        return OperatingPoint(
            rpm=rpm,
            map_kpa=manifold_pressure(env.ambient_pressure, cmd.throttle),
            air_flow_kg_s=air,
            fuel_flow_kg_s=fuel,
            fuel_flow_lph=lph,
            afr=afr,
            power_kw=power,
            bsfc_kg_kwh=bsfc,
            volumetric_efficiency=volumetric_efficiency(rpm, cmd.intake_restriction),
            brake_efficiency=brake_thermal_efficiency(rpm, afr, self.spec, cmd.combustion_quality),
            density_ratio=density_ratio(env.ambient_pressure, env.ambient_temperature),
        )

    def steady_state(self, cmd: Commands, env: Environment,
                     oil_cooling_efficiency: float = 1.0,
                     oil_leak_psi: float = 0.0,
                     alternator_field: float = 1.0) -> dict[str, float]:
        """Full steady-state instrument set for the given command point."""
        op = self.operating_point(cmd, env)
        load_fraction = min(1.2, op.power_kw / self.spec.rated_power_kw)
        cht = self.head_temperature(op.fuel_flow_kg_s, op.afr, env)
        oil_t = self.oil_temperature(cht, op.power_kw, op.rpm, env, oil_cooling_efficiency)
        egt = self.exhaust_temperature(op.fuel_flow_kg_s, op.air_flow_kg_s, op.afr, env, load_fraction)
        amps = self.alternator_output(op.rpm, alternator_field)
        return {
            'rpm': op.rpm,
            'cht': cht,
            'egt': egt,
            'oil_pressure': self.oil_pressure(op.rpm, oil_t, oil_leak_psi),
            'oil_temperature': oil_t,
            'fuel_flow': op.fuel_flow_lph,
            'fuel_flow_kg_s': op.fuel_flow_kg_s,
            'air_flow_kg_s': op.air_flow_kg_s,
            'manifold_pressure': op.map_kpa,
            'air_fuel_ratio': op.afr,
            'power_kw': op.power_kw,
            'bsfc': op.bsfc_kg_kwh,
            'volumetric_efficiency': op.volumetric_efficiency,
            'injection_timing': self.injection_advance(op.rpm, load_fraction),
            'vibration': expected_rms(healthy_drivers(load_fraction)),
            'alternator_output': amps,
            'battery_voltage': 13.9 if amps > 6.0 else 12.1 + 0.28 * amps,
            'throttle': cmd.throttle * 100.0,
            'engine_load': load_fraction * 100.0,
            'ambient_temperature': env.ambient_temperature,
            'ambient_pressure': env.ambient_pressure,
            'density_ratio': op.density_ratio,
            'pressure_altitude': pressure_altitude_m(env.ambient_pressure),
        }
