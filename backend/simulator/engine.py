"""Stateful synthetic aero piston engine built on the physics core.

Design rules that make the twin meaningful:

1. Faults perturb *physical parameters* (cooling airflow, intake restriction,
   mixture metering error, combustion quality, oil leakage), never the output
   readings directly. Symptoms therefore propagate through the same
   thermodynamic model the twin uses, so cross-channel consistency is real.
2. Sensor faults are applied on the *sensor side only*, after the physics. That
   is what lets the diagnostic layer separate a drifting CHT probe from genuine
   overheating: real overheating also moves oil temperature and EGT.
3. Thermal masses have first-order lag, so instruments trail their steady-state
   targets during transients. A naive expected-vs-actual comparison therefore
   produces false alarms in throttle transitions, which is exactly what the
   Kalman-filter twin in ``backend/digital_twin`` is there to prevent.
4. Wear accumulates from operating stress (Arrhenius-style temperature
   acceleration, cubic speed dependence), not from a fixed counter.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from backend.config.settings import FAULTS, MISSION_PROFILES, settings
from backend.simulator.physics import Commands, EnginePhysics, Environment
from backend.simulator.vibration import VibrationDrivers, VibrationSynthesiser, healthy_drivers

Fault = Literal[
    'normal', 'misfire', 'injector_abnormality', 'overheating', 'lubrication_issue',
    'excessive_vibration', 'sensor_drift', 'combustion_instability',
    'coking_degradation', 'alternator_failure',
]

# Time constants in seconds for the first-order thermal / mechanical lags.
TAU = {
    'rpm': 1.6,
    'egt': 2.5,
    'cht': 20.0,
    'oil_temperature': 65.0,
    'oil_pressure': 1.2,
}

# How quickly each fault develops from incipient to fully expressed, in seconds.
FAULT_RAMP = {
    'normal': 1.0,
    'misfire': 6.0,
    'injector_abnormality': 25.0,
    'overheating': 70.0,
    'lubrication_issue': 90.0,
    'excessive_vibration': 15.0,
    'sensor_drift': 120.0,
    'combustion_instability': 12.0,
    'coking_degradation': 240.0,
    'alternator_failure': 20.0,
}

MISSION_PRESETS: dict[str, dict[str, float]] = {
    #                       throttle  amb C  amb kPa  cooling airflow
    'normal_cruise':            {'throttle': 0.68, 'ambient_temperature': 18.0, 'ambient_pressure': 95.0, 'airspeed_ratio': 1.00},
    'high_altitude':            {'throttle': 0.92, 'ambient_temperature': -8.0, 'ambient_pressure': 68.0, 'airspeed_ratio': 1.05},
    'hot_weather':              {'throttle': 0.70, 'ambient_temperature': 44.0, 'ambient_pressure': 99.0, 'airspeed_ratio': 0.85},
    'long_endurance':           {'throttle': 0.56, 'ambient_temperature': 24.0, 'ambient_pressure': 92.0, 'airspeed_ratio': 0.95},
    'rapid_throttle_transition': {'throttle': 0.72, 'ambient_temperature': 22.0, 'ambient_pressure': 97.0, 'airspeed_ratio': 1.00},
}

# One simulated second represents this many seconds of engine life, so that
# degradation and RUL trends are observable inside a short demonstration.
WEAR_ACCELERATION = 1800.0


@dataclass
class Telemetry:
    """One acquisition frame, as it would arrive from the ECU/FADEC over CAN."""
    timestamp: str
    rpm: float
    cht: float
    egt: float
    oil_pressure: float
    oil_temperature: float
    fuel_flow: float
    vibration: float
    battery_voltage: float
    alternator_output: float
    throttle: float
    engine_load: float
    manifold_pressure: float
    air_fuel_ratio: float
    injection_timing: float
    ambient_temperature: float
    ambient_pressure: float
    airspeed_ratio: float
    density_ratio: float
    pressure_altitude: float
    power_kw: float
    engine_hours: float
    egt_spread: float
    cht_spread: float
    hottest_cylinder: int
    coldest_cylinder: int

    def dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulatorFrame:
    telemetry: Telemetry
    waveform: np.ndarray
    vibration_features: dict[str, float]
    cylinders: dict[str, list[float]]
    wear: dict[str, float]
    truth: dict[str, float | str]
    commands: dict[str, float]


class EngineSimulator:
    """Physics-driven synthetic engine with injectable, progressive faults."""

    def __init__(self, seed: int | None = None) -> None:
        self.spec = settings.engine
        self.physics = EnginePhysics(self.spec)
        self.seed = settings.seed if seed is None else seed
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.vib = VibrationSynthesiser(settings.vib_sample_rate_hz, settings.vib_block_samples, self.rng)
        self.t = 0
        self.engine_hours = 0.0
        self.running = False
        self.fault: Fault = 'normal'
        self.fault_started_at = 0
        self.mission = 'normal_cruise'
        self.throttle_command = MISSION_PRESETS['normal_cruise']['throttle']
        self.ambient_temperature = MISSION_PRESETS['normal_cruise']['ambient_temperature']
        self.ambient_pressure = MISSION_PRESETS['normal_cruise']['ambient_pressure']
        self.airspeed_ratio = MISSION_PRESETS['normal_cruise']['airspeed_ratio']
        self.wear = {'thermal': 0.0, 'lubrication': 0.0, 'combustion': 0.0,
                     'mechanical': 0.0, 'electrical': 0.0, 'fuel_delivery': 0.0}
        self.sensor_bias = {'cht': 0.0}
        self.battery_voltage = 13.9
        self.state: dict[str, float] | None = None
        n = self.spec.cylinders
        # Small permanent build tolerances so cylinders are never identical.
        self.cyl_trim = [float(x) for x in self.rng.normal(0.0, 0.22, n)]
        self.cyl_cooling_trim = [float(1.0 + x) for x in self.rng.normal(0.0, 0.015, n)]
        self.affected_cylinder = 2  # zero-based; cylinder #3 in operator numbering
        self.cyl_egt: list[float] | None = None
        self.cyl_cht: list[float] | None = None

    def configure_mission(self, mission: str) -> None:
        preset = MISSION_PRESETS.get(mission, MISSION_PRESETS['normal_cruise'])
        self.mission = mission if mission in MISSION_PROFILES else 'normal_cruise'
        self.throttle_command = preset['throttle']
        self.ambient_temperature = preset['ambient_temperature']
        self.ambient_pressure = preset['ambient_pressure']
        self.airspeed_ratio = preset['airspeed_ratio']

    def inject_fault(self, fault: str) -> None:
        if fault not in FAULTS:
            raise ValueError(f'unsupported fault: {fault}')
        if fault != self.fault:
            self.fault = fault  # type: ignore[assignment]
            self.fault_started_at = self.t
            if fault == 'normal':
                self.sensor_bias['cht'] = 0.0

    @property
    def severity(self) -> float:
        """0..1 progression of the currently injected fault."""
        if self.fault == 'normal':
            return 0.0
        elapsed = max(0, self.t - self.fault_started_at) * settings.sample_period_s
        return float(min(1.0, elapsed / FAULT_RAMP.get(self.fault, 30.0)))

    # ------------------------------------------------------------ command set
    def _throttle(self) -> float:
        """Mission throttle schedule, including the transition profile."""
        base = self.throttle_command
        if self.mission == 'rapid_throttle_transition':
            # Repeated 40 s accelerate / decelerate cycle.
            phase = (self.t * settings.sample_period_s) % 40.0
            base = 0.95 if phase < 12.0 else (0.48 if phase < 24.0 else 0.72)
        else:
            base += 0.012 * math.sin(self.t / 55.0)
        return float(np.clip(base, 0.35, 1.0))

    def _physical_inputs(self) -> tuple[Commands, Environment, dict[str, list[float]]]:
        """Commands, environment and per-cylinder deviations after fault effects."""
        sev = self.severity
        f = self.fault
        w = self.wear
        n = self.spec.cylinders
        target = self.affected_cylinder
        cyl = {
            'lean_bias': list(self.cyl_trim),
            'quality': [1.0] * n,
            'cooling': list(self.cyl_cooling_trim),
        }

        cmd = Commands(
            throttle=self._throttle(),
            prop_load=1.0,
            mixture_compensation=0.88,
            intake_restriction=0.35 * w['fuel_delivery'],
            combustion_quality=1.0 - 0.06 * w['combustion'],
            lean_bias=0.0,
        )
        env = Environment(
            ambient_temperature=self.ambient_temperature,
            ambient_pressure=self.ambient_pressure,
            airspeed_ratio=self.airspeed_ratio * (1.0 - 0.10 * w['thermal']),
        )
        self.oil_cooling_efficiency = 1.0 - 0.15 * w['lubrication']
        self.oil_leak_psi = 6.0 * w['lubrication']
        self.alternator_field = 1.0 - 0.10 * w['electrical']

        if f == 'misfire':
            # One cylinder drops out: lost work on that cylinder, its EGT falls,
            # and the unfired charge puts energy at the half crank order.
            cyl['quality'][target] = max(0.0, 1.0 - 1.05 * sev)
            cmd.combustion_quality *= 1.0 - sev / n
        elif f == 'injector_abnormality':
            # Partially clogged injector: that cylinder runs lean and its EGT rises.
            cyl['lean_bias'][target] += 3.1 * sev + 0.6 * sev * math.sin(self.t * 0.6)
            cmd.lean_bias += 0.8 * sev
            cmd.combustion_quality *= 1.0 - 0.04 * sev
        elif f == 'overheating':
            # Blocked cooling duct / failed cowl flap: real loss of cooling airflow.
            env.airspeed_ratio *= 1.0 - 0.55 * sev
            self.oil_cooling_efficiency *= 1.0 - 0.35 * sev
        elif f == 'lubrication_issue':
            self.oil_leak_psi += 22.0 * sev
            self.oil_cooling_efficiency *= 1.0 - 0.45 * sev
        elif f == 'combustion_instability':
            cmd.combustion_quality *= 1.0 - 0.10 * sev * (1.0 + math.sin(self.t * 1.15))
            cmd.lean_bias += 1.6 * sev * math.sin(self.t * 0.9)
        elif f == 'coking_degradation':
            # Carbon build-up on valves/intake: breathing loss, so VE falls.
            cmd.intake_restriction = min(0.9, cmd.intake_restriction + 0.55 * sev)
            cmd.combustion_quality *= 1.0 - 0.05 * sev
            for i in range(n):
                cyl['lean_bias'][i] += 0.6 * sev * (i / max(1, n - 1))
        elif f == 'alternator_failure':
            self.alternator_field *= 1.0 - 0.95 * sev
        elif f == 'sensor_drift':
            self.sensor_bias['cht'] = 46.0 * sev

        return cmd, env, cyl

    # ------------------------------------------------------------- vibration
    def _vibration_drivers(self, rpm: float, load_fraction: float) -> VibrationDrivers:
        sev = self.severity
        f = self.fault
        w = self.wear
        # Start from the shared healthy baseline, then add wear and fault effects.
        d = healthy_drivers(load_fraction)
        d.imbalance += 0.22 * w['mechanical']
        d.half_order += 0.05 * w['combustion']
        d.broadband += 0.18 * w['lubrication']
        if f == 'misfire':
            d.half_order += 0.42 * sev
            d.extra_orders[1.5] = 0.16 * sev
            d.firing *= 1.0 - 0.16 * sev
            d.jitter = 0.25 * sev
        elif f == 'excessive_vibration':
            d.imbalance += 0.62 * sev
            d.extra_orders[3.0] = 0.14 * sev
        elif f == 'lubrication_issue':
            d.broadband += 0.34 * sev
            d.noise += 0.03 * sev
        elif f == 'combustion_instability':
            d.am_depth = 0.38 * sev
            d.am_hz = 2.5
            d.jitter = 0.7 * sev
            d.extra_orders[1.5] = 0.10 * sev
            d.broadband += 0.12 * sev
        elif f == 'injector_abnormality':
            d.half_order += 0.10 * sev
            d.am_depth = 0.12 * sev
        elif f == 'coking_degradation':
            d.half_order += 0.05 * sev
            d.broadband += 0.05 * sev
        return d

    # ------------------------------------------------------------------- wear
    def _accumulate_wear(self, reading: dict[str, float], vib: dict[str, float]) -> None:
        """Stress-driven degradation, referenced to reaching 1.0 at TBO."""
        hours = settings.sample_period_s * WEAR_ACCELERATION / 3600.0
        self.engine_hours += hours
        base = hours / self.spec.tbo_hours

        cht = reading['cht']
        oil_t = reading['oil_temperature']
        oil_p = max(1.0, reading['oil_pressure'])
        rpm_ratio = reading['rpm'] / self.spec.max_rpm

        # Arrhenius-style acceleration: every ~28 K above reference doubles rate.
        thermal_rate = math.exp((cht - 185.0) / 28.0)
        lube_rate = math.exp((oil_t - 100.0) / 22.0) + max(0.0, (45.0 - oil_p) / 18.0) ** 2
        mech_rate = (max(0.2, rpm_ratio) ** 3) / 0.55 + 2.5 * max(0.0, vib['overall_rms'] - 0.9)
        comb_rate = 1.0 + 6.0 * max(0.0, vib['misfire_index'] - 0.10) + max(0.0, (reading['egt'] - 760.0) / 40.0)
        elec_rate = 1.0 + max(0.0, (13.2 - reading['battery_voltage'])) * 3.0
        fuel_rate = 1.0 + 2.0 * abs(reading['air_fuel_ratio'] - 14.0) / 4.0

        rates = {'thermal': thermal_rate, 'lubrication': lube_rate, 'mechanical': mech_rate,
                 'combustion': comb_rate, 'electrical': elec_rate, 'fuel_delivery': fuel_rate}
        for key, rate in rates.items():
            self.wear[key] = float(min(1.0, self.wear[key] + base * max(0.05, rate)))

    # ------------------------------------------------------------------- step
    def step(self) -> SimulatorFrame:
        self.t += 1
        dt = settings.sample_period_s
        cmd, env, cyl = self._physical_inputs()
        target = self.physics.steady_state(
            cmd, env,
            oil_cooling_efficiency=self.oil_cooling_efficiency,
            oil_leak_psi=self.oil_leak_psi,
            alternator_field=self.alternator_field,
        )

        if self.state is None:  # cold start: begin soaked at ambient
            self.state = dict(target)
            self.state['cht'] = env.ambient_temperature + 12.0
            self.state['oil_temperature'] = env.ambient_temperature + 6.0
            self.state['egt'] = env.ambient_temperature + 60.0

        for key, tau in TAU.items():
            if key == 'oil_pressure':
                continue  # handled below, it depends on the lagged oil temperature
            alpha = dt / max(dt, tau)
            self.state[key] += (target[key] - self.state[key]) * alpha
        for key, value in target.items():
            if key not in TAU:
                self.state[key] = value

        # Oil pressure follows viscosity, and viscosity follows the oil temperature
        # the engine *actually* has right now, not the temperature it is heading
        # towards. Driving it from the steady-state target instead would make
        # pressure lead temperature during a throttle change, which no real
        # lubrication system does.
        oil_pressure_target = self.physics.oil_pressure(
            self.state['rpm'], self.state['oil_temperature'], self.oil_leak_psi)
        alpha_oil = dt / max(dt, TAU['oil_pressure'])
        self.state['oil_pressure'] += (oil_pressure_target - self.state['oil_pressure']) * alpha_oil

        load_fraction = max(0.0, self.state['engine_load'] / 100.0)

        # Per-cylinder probes, lagged with the same thermal masses as the bulk values.
        egt_target, cht_target = self.physics.cylinder_temperatures(
            target['air_flow_kg_s'], target['fuel_flow_kg_s'], target['air_fuel_ratio'], env,
            load_fraction, cyl['lean_bias'], cyl['quality'], cyl['cooling'])
        if self.cyl_egt is None:
            self.cyl_egt = [env.ambient_temperature + 60.0] * self.spec.cylinders
            self.cyl_cht = [env.ambient_temperature + 12.0] * self.spec.cylinders
        a_egt, a_cht = dt / max(dt, TAU['egt']), dt / max(dt, TAU['cht'])
        self.cyl_egt = [v + (egt_target[i] - v) * a_egt for i, v in enumerate(self.cyl_egt)]
        self.cyl_cht = [v + (cht_target[i] - v) * a_cht for i, v in enumerate(self.cyl_cht)]

        waveform = self.vib.generate(self.state['rpm'], self._vibration_drivers(self.state['rpm'], load_fraction))
        from backend.signal_processing import spectral  # local import keeps module graph shallow
        vib_features = spectral.analyse(waveform, self.state['rpm'], settings.vib_sample_rate_hz)

        # Electrical bus: alternator supports the load or the battery drains.
        amps = self.state['alternator_output']
        if amps > 8.0:
            self.battery_voltage += (13.9 - self.battery_voltage) * 0.25
        else:
            self.battery_voltage -= 0.055 * (1.0 + (8.0 - amps) / 8.0)
        self.battery_voltage = float(np.clip(self.battery_voltage, 9.5, 14.4))

        reading = dict(self.state)
        reading['battery_voltage'] = self.battery_voltage
        reading['vibration'] = vib_features['overall_rms']
        # A collector probe reads the cylinder average, so keep scalar and
        # per-cylinder channels consistent by deriving one from the other.
        reading['egt'] = float(np.mean(self.cyl_egt))
        reading['cht'] = float(np.mean(self.cyl_cht))
        self.state['egt'], self.state['cht'] = reading['egt'], reading['cht']
        self._accumulate_wear(reading, vib_features)

        # Measurement chain: sensor noise, then any sensor-side fault bias.
        # Sensor scatter matching the declared accuracies in the ICD.
        noise = {'rpm': 10.0, 'cht': 2.0, 'egt': 5.0, 'oil_pressure': 0.5, 'oil_temperature': 1.0,
                 'fuel_flow': 0.28, 'battery_voltage': 0.05, 'alternator_output': 0.30,
                 'manifold_pressure': 0.5, 'injection_timing': 0.30, 'air_fuel_ratio': 0.15}
        measured = dict(reading)
        for key, sd in noise.items():
            measured[key] = reading[key] + float(self.rng.normal(0.0, sd))
        for key, bias in self.sensor_bias.items():
            measured[key] += bias

        egt_cyl = [round(v + float(self.rng.normal(0.0, 5.0)), 1) for v in self.cyl_egt]
        cht_cyl = [round(v + float(self.rng.normal(0.0, 2.0)) + self.sensor_bias['cht'], 1)
                   for v in self.cyl_cht]
        hottest = int(max(range(len(egt_cyl)), key=lambda i: egt_cyl[i])) + 1
        coldest = int(min(range(len(egt_cyl)), key=lambda i: egt_cyl[i])) + 1

        telemetry = Telemetry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            rpm=round(measured['rpm'], 1),
            cht=round(measured['cht'], 2),
            egt=round(measured['egt'], 2),
            oil_pressure=round(max(0.0, measured['oil_pressure']), 2),
            oil_temperature=round(measured['oil_temperature'], 2),
            fuel_flow=round(max(0.0, measured['fuel_flow']), 3),
            vibration=round(vib_features['overall_rms'], 4),
            battery_voltage=round(measured['battery_voltage'], 2),
            alternator_output=round(max(0.0, measured['alternator_output']), 2),
            throttle=round(measured['throttle'], 2),
            engine_load=round(measured['engine_load'], 2),
            manifold_pressure=round(measured['manifold_pressure'], 2),
            air_fuel_ratio=round(measured['air_fuel_ratio'], 2),
            injection_timing=round(measured['injection_timing'], 2),
            ambient_temperature=round(env.ambient_temperature, 2),
            ambient_pressure=round(env.ambient_pressure, 2),
            airspeed_ratio=round(self.airspeed_ratio, 3),
            density_ratio=round(measured['density_ratio'], 4),
            pressure_altitude=round(measured['pressure_altitude'], 1),
            power_kw=round(measured['power_kw'], 2),
            engine_hours=round(self.engine_hours, 2),
            egt_spread=round(max(egt_cyl) - min(egt_cyl), 1),
            cht_spread=round(max(cht_cyl) - min(cht_cyl), 1),
            hottest_cylinder=hottest,
            coldest_cylinder=coldest,
        )
        return SimulatorFrame(
            telemetry=telemetry,
            waveform=waveform,
            vibration_features=vib_features,
            cylinders={'egt': egt_cyl, 'cht': cht_cyl},
            wear=dict(self.wear),
            truth={'fault': self.fault, 'severity': round(self.severity, 3), 'mission': self.mission},
            commands={'throttle': cmd.throttle, 'intake_restriction': cmd.intake_restriction,
                      'lean_bias': cmd.lean_bias, 'combustion_quality': cmd.combustion_quality},
        )
