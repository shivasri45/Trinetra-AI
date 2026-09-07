"""Remaining useful life by degradation-trend extrapolation.

RUL is not a scaled health number. Each subsystem's smoothed health index is
regressed against accumulated engine hours, and the fitted trend is extrapolated
to the failure threshold:

    RUL = (health_now - health_failure) / -slope

The slope's standard error gives a confidence interval, so the estimate is
reported as a band rather than a single misleading figure. The overall RUL is set
by the limiting subsystem, and everything is additionally capped by the hours
remaining to the published time between overhauls.

A minimum observation window is enforced. Before it is reached the estimator
reports that it is still establishing a trend instead of inventing a number.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from backend.config.settings import SUBSYSTEMS, settings

MIN_SAMPLES = 45            # minimum points before a trend is published
TREND_WINDOW = 240          # samples used for the regression
SMOOTHING = 0.12            # EMA factor applied to the noisy health index
MIN_SLOPE = 1e-4            # health per hour below which the trend is flat
CONFIDENCE_Z = 1.96         # 95 % interval

ACTIONS: dict[str, str] = {
    'thermal': 'Inspect cooling ducting, baffles and cowl flap travel; verify CHT probe calibration.',
    'lubrication': 'Sample and analyse oil, check pump output, relief valve and for external leakage.',
    'combustion': 'Borescope cylinders, check ignition harness, plugs and injector spray pattern.',
    'mechanical': 'Balance check on propeller and rotating assembly; inspect mounts and bearings.',
    'electrical': 'Test alternator output, regulator setpoint, battery capacity and bus wiring.',
    'fuel_delivery': 'Clean intake tract and valves, flow-test injectors, replace air filter.',
}


@dataclass
class SubsystemTrend:
    subsystem: str
    health: float
    slope_per_hour: float
    rul_hours: float
    rul_low: float
    rul_high: float
    r_squared: float
    samples: int
    status: str


class RulEstimator:
    """Tracks per-subsystem degradation trends and extrapolates to threshold."""

    def __init__(self) -> None:
        self.hours: deque[float] = deque(maxlen=TREND_WINDOW)
        self.history: dict[str, deque[float]] = {s: deque(maxlen=TREND_WINDOW) for s in SUBSYSTEMS}
        self.smoothed: dict[str, float] = {}

    def reset(self) -> None:
        self.hours.clear()
        for series in self.history.values():
            series.clear()
        self.smoothed.clear()

    # ------------------------------------------------------------------ update
    def update(self, engine_hours: float, twin: dict) -> dict:
        subsystems = twin.get('subsystems', {})
        self.hours.append(float(engine_hours))
        for name in SUBSYSTEMS:
            value = float(subsystems.get(name, {}).get('health', 100.0))
            previous = self.smoothed.get(name, value)
            current = previous + SMOOTHING * (value - previous)
            self.smoothed[name] = current
            self.history[name].append(current)

        tbo_remaining = max(0.0, settings.engine.tbo_hours - float(engine_hours))
        trends = [self._trend(name, tbo_remaining) for name in SUBSYSTEMS]
        ranked = sorted(trends, key=lambda t: t.rul_hours)
        limiting = ranked[0]

        established = [t for t in trends if t.status == 'trend_established']
        overall_status = 'trend_established' if established else 'establishing_trend'

        rul_hours = min(limiting.rul_hours, tbo_remaining)
        percentage = 100.0 * rul_hours / max(1e-6, settings.engine.tbo_hours)

        return {
            'status': overall_status,
            'method': 'least-squares degradation trend extrapolated to failure threshold',
            'failure_threshold_health': settings.failure_health,
            'engine_hours': round(float(engine_hours), 2),
            'tbo_hours': settings.engine.tbo_hours,
            'tbo_remaining_hours': round(tbo_remaining, 1),
            'hours': round(rul_hours, 1),
            'percentage': round(min(100.0, percentage), 1),
            'confidence_interval_hours': [round(min(limiting.rul_low, tbo_remaining), 1),
                                          round(min(limiting.rul_high, tbo_remaining), 1)],
            'limiting_subsystem': limiting.subsystem,
            'limiting_slope_per_hour': round(limiting.slope_per_hour, 5),
            'subsystems': {t.subsystem: {
                'health': round(t.health, 1),
                'slope_per_hour': round(t.slope_per_hour, 5),
                'rul_hours': round(min(t.rul_hours, tbo_remaining), 1),
                'confidence_interval_hours': [round(min(t.rul_low, tbo_remaining), 1),
                                              round(min(t.rul_high, tbo_remaining), 1)],
                'r_squared': round(t.r_squared, 3),
                'samples': t.samples,
                'status': t.status,
            } for t in trends},
            'advisory': self._advisory(limiting, rul_hours, tbo_remaining),
            'label': 'Simulated prototype RUL - not airworthiness data',
        }

    # --------------------------------------------------------------- internals
    def _trend(self, name: str, tbo_remaining: float) -> SubsystemTrend:
        series = self.history[name]
        health = series[-1] if series else 100.0
        n = len(series)
        if n < MIN_SAMPLES:
            return SubsystemTrend(name, health, 0.0, tbo_remaining, tbo_remaining, tbo_remaining,
                                  0.0, n, 'establishing_trend')

        x = np.array(list(self.hours)[-n:], dtype=float)
        y = np.array(list(series), dtype=float)
        if float(np.ptp(x)) < 1e-9:
            return SubsystemTrend(name, health, 0.0, tbo_remaining, tbo_remaining, tbo_remaining,
                                  0.0, n, 'establishing_trend')

        x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
        sxx = float(np.sum((x - x_mean) ** 2))
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
        intercept = y_mean - slope * x_mean
        fitted = intercept + slope * x
        residual = y - fitted
        sse = float(np.sum(residual ** 2))
        sst = float(np.sum((y - y_mean) ** 2))
        r_squared = 1.0 - sse / sst if sst > 1e-12 else 0.0
        slope_se = math.sqrt(sse / max(1, n - 2) / sxx) if sxx > 0 else 0.0

        margin = health - settings.failure_health
        if slope >= -MIN_SLOPE or margin <= 0.0:
            status = 'no_degradation_trend' if margin > 0.0 else 'below_threshold'
            rul = tbo_remaining if margin > 0.0 else 0.0
            return SubsystemTrend(name, health, slope, rul, rul, rul, r_squared, n, status)

        rul = margin / -slope
        # First-order propagation of the slope uncertainty into the RUL.
        spread = CONFIDENCE_Z * rul * (slope_se / abs(slope)) if slope_se > 0 else 0.0
        low = max(0.0, rul - spread)
        high = rul + spread if spread > 0 else rul
        return SubsystemTrend(name, health, slope, rul, low, min(high, tbo_remaining),
                              r_squared, n, 'trend_established')

    @staticmethod
    def _advisory(limiting: SubsystemTrend, rul_hours: float, tbo_remaining: float) -> dict:
        if limiting.status == 'establishing_trend':
            return {'priority': 'MONITOR', 'due_hours': None,
                    'action': 'Continue monitoring; degradation trend not yet established.'}
        if rul_hours >= tbo_remaining - 1e-6:
            return {'priority': 'SCHEDULED', 'due_hours': round(tbo_remaining, 1),
                    'action': 'No accelerated degradation detected; overhaul at published TBO.'}
        if rul_hours < 10.0:
            priority = 'GROUND'
        elif rul_hours < 50.0:
            priority = 'IMMEDIATE'
        elif rul_hours < 200.0:
            priority = 'PRIORITY'
        else:
            priority = 'SCHEDULED'
        return {
            'priority': priority,
            'due_hours': round(rul_hours, 1),
            'subsystem': limiting.subsystem,
            'action': ACTIONS.get(limiting.subsystem, 'Perform a general engine inspection.'),
        }
