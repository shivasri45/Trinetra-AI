"""The digital twin: a synchronised virtual engine and its health assessment.

The twin is given only what a real ground station would receive - the measured
telemetry frame. From the measured throttle and atmospheric conditions it solves
the same thermodynamic model the physical engine obeys, runs a Kalman filter over
every channel to obtain a transient-aware estimate, and converts the normalised
innovations into subsystem health indices.

It has no knowledge of which fault was injected, nor of the simulator's internal
wear state. Everything it reports is derived from the telemetry frame alone.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from backend.config.settings import SUBSYSTEMS, settings
from backend.digital_twin.estimator import StateEstimator
from backend.simulator.physics import Commands, EnginePhysics, Environment

# Which normalised innovations feed which subsystem health index, and how hard.
# Which normalised innovations feed which subsystem, how hard, and in which
# direction. Direction matters: a cylinder running *cold* is a combustion problem,
# not a thermal one, so the thermal index only reacts to over-temperature.
#   +1 -> only positive deviations count   -1 -> only negative   0 -> absolute
SUBSYSTEM_CHANNELS: dict[str, dict[str, tuple[float, int]]] = {
    'thermal': {'cht': (1.0, +1), 'egt': (0.35, +1), 'oil_temperature': (0.75, +1)},
    # Driven by the temperature-conditioned pressure residual, so that overheating
    # is not blamed on the oil system for a pressure drop it fully explains. Oil
    # temperature still carries a small weight because genuinely hot oil is a
    # lubrication concern in its own right.
    'lubrication': {'oil_pressure_cond': (1.0, -1), 'oil_temperature': (0.15, +1)},
    'combustion': {'egt': (0.8, 0), 'air_fuel_ratio': (0.6, 0), 'rpm': (0.4, -1)},
    'mechanical': {'rpm': (0.5, 0), 'vibration': (1.0, +1)},
    'electrical': {'battery_voltage': (1.0, -1), 'alternator_output': (0.8, -1)},
    'fuel_delivery': {'fuel_flow': (1.0, 0), 'air_fuel_ratio': (0.7, 0),
                      'manifold_pressure': (0.4, 0)},
}

# Per-cylinder probes roll up: over-temperature into thermal, any deviation into
# combustion (a dead cylinder shows up as a cold probe).
CYLINDER_ROLLUP: dict[str, tuple[str, int]] = {
    'thermal': ('cht_cyl', +1),
    'combustion': ('egt_cyl', 0),
}

# Extra condition indicators, in natural units, that also drive health.
INDICATOR_LIMITS: dict[str, tuple[str, float, float]] = {
    # indicator -> (subsystem, nominal, limit)
    'misfire_index': ('combustion', 0.12, 0.55),
    'imbalance_index': ('mechanical', 0.22, 0.80),
    'roughness_index': ('lubrication', 0.16, 0.70),
    'egt_spread': ('combustion', 45.0, 200.0),
    'cht_spread': ('thermal', 12.0, 60.0),
}

# Analytical redundancy: channels inside a group are physically coupled, so a
# deviation on one member alone is far more likely to be an instrumentation
# fault than a real engine fault. This is what separates a drifting CHT probe
# from genuine overheating, which necessarily moves oil temperature too.
COUPLED_GROUPS: dict[str, tuple[str, ...]] = {
    # Head and oil temperature are driven by the same heat rejection path, so they
    # move together in any genuine thermal event.
    'thermal_chain': ('cht', 'oil_temperature'),
    # Exhaust temperature is set by mixture and fuel rate, not by oil temperature.
    'combustion_chain': ('egt', 'fuel_flow', 'air_fuel_ratio'),
    'load_chain': ('rpm', 'manifold_pressure', 'fuel_flow'),
    'electrical_bus': ('alternator_output', 'battery_voltage'),
    'lubrication_chain': ('oil_pressure', 'oil_temperature'),
}
ISOLATION_THRESHOLD = 2.5

# Normalised innovation squared is a filter-consistency statistic, so it is only
# meaningful for channels whose process model is unbiased. The nominal oil
# pressure channel predicts from the *steady-state* oil temperature, which an
# engine on a 40 s throttle cycle never actually reaches, so it carries a known
# transient bias by construction. Including it swamps the aggregate: measured on
# the transition profile it contributes a mean NIS of 175 while every other
# channel stays below 1.1. The temperature-conditioned channel is the unbiased
# one and is what the statistic uses instead. The nominal channel is still
# reported, and still serves as a model feature, because its transient signature
# is informative in its own right.
NIS_EXCLUDED: frozenset[str] = frozenset({'oil_pressure'})

PENALTY_PER_SIGMA = 9.0
MAX_CHANNEL_SIGMA = 6.0
# Innovations inside this many standard deviations are statistically expected
# noise and must not consume health, otherwise a healthy engine never reads 100.
DEAD_BAND_SIGMA = 1.0


class DigitalTwin:
    def __init__(self) -> None:
        self.physics = EnginePhysics(settings.engine)
        self.estimator = StateEstimator()
        self.cylinders = settings.engine.cylinders

    # ------------------------------------------------------------- expectation
    def expected_state(self, telemetry: dict) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Nominal steady state for the measured command point.

        Nominal means a healthy engine: no intake restriction, no metering error,
        clean combustion, full cooling airflow, no oil leakage.
        """
        cmd = Commands(
            throttle=max(0.0, min(1.0, telemetry['throttle'] / 100.0)),
            prop_load=1.0,
            mixture_compensation=0.88,
            intake_restriction=0.0,
            combustion_quality=1.0,
            lean_bias=0.0,
        )
        env = Environment(
            ambient_temperature=telemetry['ambient_temperature'],
            ambient_pressure=telemetry['ambient_pressure'],
            airspeed_ratio=telemetry.get('airspeed_ratio', 1.0),
        )
        state = self.physics.steady_state(cmd, env)
        # Cascaded residual: what the pump should deliver at the *measured* oil
        # temperature and speed, rather than at the nominal expected ones.
        state['oil_pressure_cond'] = self.physics.oil_pressure(
            float(telemetry.get('rpm', state['rpm'])),
            float(telemetry.get('oil_temperature', state['oil_temperature'])))
        load_fraction = min(1.2, state['power_kw'] / settings.engine.rated_power_kw)
        egt, cht = self.physics.cylinder_temperatures(
            state['air_flow_kg_s'], state['fuel_flow_kg_s'], state['air_fuel_ratio'], env,
            load_fraction, [0.0] * self.cylinders, [1.0] * self.cylinders, [1.0] * self.cylinders)
        return state, {'egt': egt, 'cht': cht}

    # ------------------------------------------------------------------- state
    def state(self, telemetry: dict, cylinders: dict[str, list[float]],
              indicators: dict[str, float]) -> dict:
        expected, cyl_expected = self.expected_state(telemetry)
        # The conditional channel re-uses the same oil pressure measurement.
        measured = dict(telemetry)
        measured['oil_pressure_cond'] = telemetry.get('oil_pressure', 0.0)
        estimates = self.estimator.step(measured, expected, cylinders, cyl_expected)

        normalised = {k: round(e.normalised, 3) for k, e in estimates.items()}
        residuals = {k: round(e.innovation, 3) for k, e in estimates.items()}
        drift = {k: round(e.baseline_sigma, 3) for k, e in estimates.items()}
        estimate_values = {k: round(e.estimate, 3) for k, e in estimates.items()}

        # --- statistical anomaly gate: mean normalised innovation squared
        nis_values = [e.nis for name, e in estimates.items() if name not in NIS_EXCLUDED]
        mean_nis = float(np.mean(nis_values)) if nis_values else 0.0
        peak_nis = float(np.max(nis_values)) if nis_values else 0.0

        indicator_scores = self._indicator_scores(telemetry, indicators)
        validation = self._sensor_validation(normalised, drift)
        subsystems = self._subsystem_health(normalised, drift, indicator_scores)

        values = [v['health'] for v in subsystems.values()]
        overall = 0.55 * min(values) + 0.45 * float(np.mean(values))
        degradation = float(np.mean([v['degradation'] for v in subsystems.values()]))

        alert = ('NORMAL' if overall > settings.warning_health
                 else 'WARNING' if overall > settings.critical_health else 'CRITICAL')

        abnormal = sorted(
            (k for k, v in normalised.items() if abs(v) > 3.0),
            key=lambda k: -abs(normalised[k]),
        )
        return {
            'expected': {k: round(v, 3) for k, v in expected.items()},
            'estimated': estimate_values,
            'measured': {k: v for k, v in measured.items() if k in estimates},
            'residuals': residuals,
            'normalised_residuals': normalised,
            'baseline_drift': drift,
            'cylinders': {'measured': cylinders,
                          'expected': {k: [round(x, 1) for x in v] for k, v in cyl_expected.items()}},
            'condition_indicators': indicator_scores,
            'sensor_validation': validation,
            'subsystems': subsystems,
            'health_score': round(overall, 1),
            'degradation_score': round(degradation * 100.0, 1),
            'mean_nis': round(mean_nis, 2),
            'peak_nis': round(peak_nis, 2),
            'nis_channels': len(nis_values),
            'nis_excluded_channels': sorted(NIS_EXCLUDED),
            'statistical_alarm': mean_nis > settings.nis_warning,
            'operating_state': alert,
            'abnormality_indicators': abnormal[:6],
            'estimator_detail': {k: {kk: round(vv, 4) for kk, vv in asdict(e).items() if kk != 'channel'}
                                 for k, e in estimates.items()},
        }

    # --------------------------------------------------------------- internals
    @staticmethod
    def _excess(value: float, direction: int = 0) -> float:
        """Sigma beyond the dead band, optionally only in one direction."""
        if direction > 0:
            magnitude = max(0.0, value)
        elif direction < 0:
            magnitude = max(0.0, -value)
        else:
            magnitude = abs(value)
        return min(MAX_CHANNEL_SIGMA, max(0.0, magnitude - DEAD_BAND_SIGMA))

    def _sensor_validation(self, normalised: dict[str, float], drift: dict[str, float]) -> dict:
        """Flag channels that deviate in isolation from their coupled group.

        For each physically coupled group the median deviation is taken as the
        group consensus. A member that sits far above that consensus is
        inconsistent with the physics of the group, which points at the
        instrument rather than at the engine.
        """
        suspects: list[dict] = []
        groups: dict[str, dict] = {}
        isolation: dict[str, float] = {}
        for group, channels in COUPLED_GROUPS.items():
            members = {c: abs(normalised.get(c, 0.0)) + abs(drift.get(c, 0.0))
                       for c in channels if c in normalised}
            if len(members) < 2:
                continue
            consensus = float(np.median(list(members.values())))
            groups[group] = {'consensus': round(consensus, 2),
                             'members': {k: round(v, 2) for k, v in members.items()}}
            for channel, value in members.items():
                gap = value - consensus
                isolation[channel] = max(isolation.get(channel, 0.0), gap)
                if gap > ISOLATION_THRESHOLD:
                    suspects.append({'channel': channel, 'group': group,
                                     'isolation_sigma': round(gap, 2),
                                     'group_consensus_sigma': round(consensus, 2)})
        suspects.sort(key=lambda s: -s['isolation_sigma'])
        return {
            'groups': groups,
            'isolation': {k: round(v, 3) for k, v in isolation.items()},
            'suspect_channels': suspects,
            'plausibility': 'SENSOR_SUSPECT' if suspects else 'CONSISTENT',
        }

    def _indicator_scores(self, telemetry: dict, indicators: dict[str, float]) -> dict[str, dict]:
        """Normalise engineering condition indicators onto a 0..1 exceedance scale."""
        merged = dict(indicators)
        merged['egt_spread'] = telemetry.get('egt_spread', 0.0)
        merged['cht_spread'] = telemetry.get('cht_spread', 0.0)
        out: dict[str, dict] = {}
        for name, (subsystem, nominal, limit) in INDICATOR_LIMITS.items():
            value = float(merged.get(name, 0.0))
            span = max(1e-6, limit - nominal)
            exceedance = max(0.0, min(2.0, (value - nominal) / span))
            out[name] = {'value': round(value, 4), 'nominal': nominal, 'limit': limit,
                         'subsystem': subsystem, 'exceedance': round(exceedance, 3)}
        return out

    def _subsystem_health(self, normalised: dict[str, float], drift: dict[str, float],
                          indicators: dict[str, dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for subsystem in SUBSYSTEMS:
            weights = SUBSYSTEM_CHANNELS.get(subsystem, {})
            acute = 0.0
            chronic = 0.0
            for channel, (weight, direction) in weights.items():
                acute += weight * self._excess(normalised.get(channel, 0.0), direction)
                chronic += weight * self._excess(drift.get(channel, 0.0), direction)
            rollup = CYLINDER_ROLLUP.get(subsystem)
            if rollup:
                prefix, direction = rollup
                cyl = [self._excess(v, direction) for k, v in normalised.items()
                       if k.startswith(prefix)]
                if cyl:
                    acute += 0.5 * max(cyl)

            indicator_exceedance = sum(
                v['exceedance'] for v in indicators.values() if v['subsystem'] == subsystem)

            weight_total = sum(weight for weight, _ in weights.values()) or 1.0
            acute_penalty = min(60.0, PENALTY_PER_SIGMA * acute / weight_total * 1.6)
            chronic_penalty = min(35.0, PENALTY_PER_SIGMA * chronic / weight_total * 1.1)
            indicator_penalty = min(45.0, 26.0 * indicator_exceedance)

            health = max(0.0, 100.0 - acute_penalty - chronic_penalty - indicator_penalty)
            out[subsystem] = {
                'health': round(health, 1),
                'acute_penalty': round(acute_penalty, 1),
                'chronic_penalty': round(chronic_penalty, 1),
                'indicator_penalty': round(indicator_penalty, 1),
                'degradation': round(min(1.0, (chronic_penalty / 35.0) * 0.6
                                        + (indicator_penalty / 45.0) * 0.4), 4),
            }
        return out
