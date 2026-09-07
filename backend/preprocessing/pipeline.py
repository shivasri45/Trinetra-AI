"""Feature construction shared by model training and live inference.

The same code path is used offline and online, which is the only reliable way to
avoid train/serve skew. Two rules are enforced here:

1. No label, no injected-fault identifier, no simulator wear state and no fault
   severity ever enters this module. The only inputs are the telemetry frame, the
   twin's statistical residuals and the vibration spectrum - all of which a real
   ground station would have.
2. Features are dimensionless wherever possible (normalised innovations, crank
   orders referenced to the firing order, ratios against expectation), so a model
   trained at one operating point remains valid across the flight envelope.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from backend.config.settings import settings

# Channels carried through the residual feature block.
RESIDUAL_CHANNELS: tuple[str, ...] = (
    'rpm', 'cht', 'egt', 'oil_pressure', 'oil_pressure_cond', 'oil_temperature',
    'fuel_flow', 'air_fuel_ratio', 'manifold_pressure', 'vibration',
    'battery_voltage', 'alternator_output', 'injection_timing',
)

# Engineering scale used to normalise rate-of-change terms, per second.
ROC_SCALE: dict[str, float] = {
    'rpm': 120.0, 'cht': 3.0, 'egt': 12.0, 'oil_pressure': 2.5,
    'oil_pressure_cond': 2.5, 'oil_temperature': 1.0,
    'fuel_flow': 0.8, 'air_fuel_ratio': 0.4, 'manifold_pressure': 3.0, 'vibration': 0.08,
    'battery_voltage': 0.2, 'alternator_output': 1.0, 'injection_timing': 1.0,
}

VIBRATION_FEATURES: tuple[str, ...] = (
    'overall_rms', 'crest_factor', 'spectral_kurtosis', 'misfire_index',
    'imbalance_index', 'roughness_index',
)
ORDER_RATIOS: tuple[str, ...] = ('order_0p5', 'order_1p0', 'order_1p5', 'order_3p0', 'order_4p0')

RELATIVE_CHANNELS: tuple[str, ...] = ('rpm', 'cht', 'egt', 'oil_pressure', 'fuel_flow', 'power_kw')

# Analytical-redundancy isolation scores from the twin's coupled-group check.
ISOLATION_CHANNELS: tuple[str, ...] = (
    'cht', 'egt', 'oil_temperature', 'oil_pressure', 'rpm', 'fuel_flow',
    'manifold_pressure', 'alternator_output', 'battery_voltage',
)


def feature_names() -> list[str]:
    """Canonical, stable feature ordering."""
    names: list[str] = []
    for channel in RESIDUAL_CHANNELS:
        names += [f'{channel}_nres', f'{channel}_drift', f'{channel}_nres_std', f'{channel}_roc']
    names += [f'vib_{name}' for name in VIBRATION_FEATURES]
    names += [f'vib_{name}_ratio' for name in ORDER_RATIOS]
    names += ['egt_spread_rel', 'cht_spread_rel', 'egt_cyl_max_dev', 'egt_cyl_min_dev',
              'cht_cyl_max_dev', 'egt_cyl_nres_max', 'cht_cyl_nres_max']
    names += ['rpm_frac', 'load_frac', 'throttle_frac', 'density_ratio', 'ambient_norm', 'altitude_norm']
    names += [f'{channel}_rel' for channel in RELATIVE_CHANNELS]
    names += [f'{channel}_isolation' for channel in ISOLATION_CHANNELS]
    names += ['isolation_max', 'isolation_count']
    return names


FEATURE_NAMES: tuple[str, ...] = tuple(feature_names())


class FeatureBuilder:
    """Stateful windowed feature extractor."""

    def __init__(self, window: int = 15) -> None:
        self.window = window
        self.nres_history: deque[dict[str, float]] = deque(maxlen=window)
        self.value_history: deque[dict[str, float]] = deque(maxlen=window)

    def reset(self) -> None:
        self.nres_history.clear()
        self.value_history.clear()

    def transform(self, telemetry: dict, twin: dict, vibration: dict,
                  cylinders: dict[str, list[float]]) -> dict[str, float]:
        nres = twin['normalised_residuals']
        drift = twin['baseline_drift']
        expected = twin['expected']

        # The twin's ``measured`` block includes derived channels such as the
        # temperature-conditioned oil pressure, which the raw frame does not carry.
        readings = {**telemetry, **twin.get('measured', {})}
        self.nres_history.append({k: float(nres.get(k, 0.0)) for k in RESIDUAL_CHANNELS})
        self.value_history.append({k: float(readings.get(k, 0.0)) for k in RESIDUAL_CHANNELS})

        out: dict[str, float] = {}
        dt = settings.sample_period_s
        for channel in RESIDUAL_CHANNELS:
            series = [row[channel] for row in self.nres_history]
            values = [row[channel] for row in self.value_history]
            out[f'{channel}_nres'] = float(nres.get(channel, 0.0))
            out[f'{channel}_drift'] = float(drift.get(channel, 0.0))
            out[f'{channel}_nres_std'] = float(np.std(series)) if len(series) > 2 else 0.0
            roc = (values[-1] - values[-2]) / dt if len(values) > 1 else 0.0
            out[f'{channel}_roc'] = roc / ROC_SCALE.get(channel, 1.0)

        for name in VIBRATION_FEATURES:
            out[f'vib_{name}'] = float(vibration.get(name, 0.0))
        firing = max(1e-6, float(vibration.get('order_2p0', 0.0)))
        for name in ORDER_RATIOS:
            out[f'vib_{name}_ratio'] = float(vibration.get(name, 0.0)) / firing

        egt_cyl = cylinders.get('egt', []) or [0.0]
        cht_cyl = cylinders.get('cht', []) or [0.0]
        egt_mean = float(np.mean(egt_cyl)) or 1.0
        cht_mean = float(np.mean(cht_cyl)) or 1.0
        out['egt_spread_rel'] = (max(egt_cyl) - min(egt_cyl)) / abs(egt_mean)
        out['cht_spread_rel'] = (max(cht_cyl) - min(cht_cyl)) / abs(cht_mean)
        out['egt_cyl_max_dev'] = (max(egt_cyl) - egt_mean) / abs(egt_mean)
        out['egt_cyl_min_dev'] = (min(egt_cyl) - egt_mean) / abs(egt_mean)
        out['cht_cyl_max_dev'] = (max(cht_cyl) - cht_mean) / abs(cht_mean)
        cyl_egt_nres = [abs(v) for k, v in nres.items() if k.startswith('egt_cyl')]
        cyl_cht_nres = [abs(v) for k, v in nres.items() if k.startswith('cht_cyl')]
        out['egt_cyl_nres_max'] = max(cyl_egt_nres) if cyl_egt_nres else 0.0
        out['cht_cyl_nres_max'] = max(cyl_cht_nres) if cyl_cht_nres else 0.0

        out['rpm_frac'] = float(telemetry.get('rpm', 0.0)) / settings.engine.max_rpm
        out['load_frac'] = float(telemetry.get('engine_load', 0.0)) / 100.0
        out['throttle_frac'] = float(telemetry.get('throttle', 0.0)) / 100.0
        out['density_ratio'] = float(telemetry.get('density_ratio', 1.0))
        out['ambient_norm'] = (float(telemetry.get('ambient_temperature', 15.0)) - 15.0) / 30.0
        out['altitude_norm'] = float(telemetry.get('pressure_altitude', 0.0)) / 6000.0

        for channel in RELATIVE_CHANNELS:
            exp = float(expected.get(channel, 0.0))
            act = float(telemetry.get(channel, 0.0))
            out[f'{channel}_rel'] = (act - exp) / abs(exp) if abs(exp) > 1e-6 else 0.0

        validation = twin.get('sensor_validation', {})
        isolation = validation.get('isolation', {})
        for channel in ISOLATION_CHANNELS:
            out[f'{channel}_isolation'] = float(isolation.get(channel, 0.0))
        out['isolation_max'] = max(isolation.values()) if isolation else 0.0
        out['isolation_count'] = float(len(validation.get('suspect_channels', [])))

        return out

    @staticmethod
    def to_vector(features: dict[str, float]) -> np.ndarray:
        return np.array([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)
