"""Physics-informed state estimation for the digital twin.

Each monitored channel is tracked by a scalar Kalman filter whose process model
is the engine's own first-order thermal/mechanical lag:

    x[k+1] = x[k] + (u[k] - x[k]) * dt/tau + w,     w ~ N(0, Q)
    z[k]   = x[k] + v,                              v ~ N(0, R)

``u`` is the steady-state target produced by the thermodynamic model from the
*measured* throttle and atmospheric conditions, so the filter knows where the
engine is heading, while ``tau`` tells it how fast it is allowed to get there.

Why this matters: comparing a raw measurement against a steady-state expectation
raises false alarms during every throttle transient, because a cylinder head
needs tens of seconds to catch up. The filter's innovation (measurement minus
one-step prediction) is transient-aware, and dividing it by the innovation
standard deviation gives a statistically normalised residual instead of an
arbitrary hand-tuned scale factor.

The filter additionally maintains a slowly adapting per-channel bias. Gradual
drift is absorbed into that bias and reported as *degradation*, while whatever
remains in the innovation is reported as an *anomaly*. That separation is what
lets a slow coking trend be distinguished from an abrupt misfire.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from backend.config.settings import settings

# tau: seconds; q: process variance per step; r: sensor variance.
#
# The r values are the squared 1-sigma accuracies declared in the interface
# control document (docs/interface_control.md) for each transducer. Getting these
# right matters: an over-optimistic sensor model makes ordinary instrument scatter
# and normal wear look like faults, which is the usual cause of nuisance alarms.
CHANNEL_MODELS: dict[str, dict[str, float]] = {
    'rpm':               {'tau': 1.6, 'q': 900.0, 'r': 100.0},    # +/- 10 rpm
    'cht':               {'tau': 20.0, 'q': 0.55, 'r': 4.0},      # +/- 2 C  (type-K)
    'egt':               {'tau': 2.5, 'q': 14.0, 'r': 25.0},      # +/- 5 C
    'oil_pressure':      {'tau': 1.2, 'q': 0.35, 'r': 0.25},      # +/- 0.5 psi
    # Same measurement, but compared against the pressure the pump should deliver
    # at the *measured* oil temperature. Isolates a genuine lubrication fault from
    # the pressure loss that hot, thin oil legitimately causes.
    #
    # The process noise is deliberately much larger than the sensor noise. The
    # viscosity correlation behind this expectation is fitted around normal
    # operating temperature, so extrapolating it to 150 C and beyond carries real
    # model uncertainty. Declaring that uncertainty here keeps an extrapolation
    # artefact from reading as a 7-sigma lubrication fault.
    'oil_pressure_cond': {'tau': 1.2, 'q': 2.5, 'r': 0.25},
    'oil_temperature':   {'tau': 65.0, 'q': 0.05, 'r': 1.0},      # +/- 1 C
    'fuel_flow':         {'tau': 1.0, 'q': 0.03, 'r': 0.09},      # +/- 2 % of reading
    'manifold_pressure': {'tau': 0.5, 'q': 0.30, 'r': 0.25},      # +/- 0.5 kPa
    'air_fuel_ratio':    {'tau': 1.0, 'q': 0.02, 'r': 0.0225},    # +/- 0.15 AFR
    'injection_timing':  {'tau': 0.5, 'q': 0.10, 'r': 0.09},      # +/- 0.3 deg
    'alternator_output': {'tau': 1.0, 'q': 0.20, 'r': 0.09},      # +/- 0.3 A
    'battery_voltage':   {'tau': 4.0, 'q': 0.004, 'r': 0.0025},   # +/- 0.05 V
    'vibration':         {'tau': 1.5, 'q': 0.004, 'r': 0.0016},   # +/- 0.04 g
}

BASELINE_RATE = 1.0 / 400.0      # slow bias adaptation
BASELINE_GATE = 4.0              # freeze adaptation while grossly abnormal
# Build tolerance, calibration offset and gradual wear are absorbed here and
# reported as degradation. Anything beyond this stays visible as innovation so it
# is diagnosed as a fault rather than quietly normalised away.
BASELINE_LIMIT = 2.5             # max learned bias, in sigma


@dataclass
class ChannelEstimate:
    channel: str
    measurement: float
    prediction: float
    estimate: float
    steady_state: float
    innovation: float
    innovation_sigma: float
    normalised: float          # innovation / sigma, baseline removed
    nis: float                 # normalised innovation squared
    baseline_bias: float       # learned slow drift, engineering units
    baseline_sigma: float      # learned slow drift, in sigma


class ChannelFilter:
    """Scalar Kalman filter with a first-order lag process model."""

    def __init__(self, channel: str, tau: float, q: float, r: float, dt: float) -> None:
        self.channel = channel
        self.alpha = dt / max(dt, tau)
        self.q = q
        self.r = r
        self.x: float | None = None
        self.p = r * 4.0
        self.baseline = 0.0

    def step(self, steady_state: float, measurement: float) -> ChannelEstimate:
        if self.x is None:
            self.x = measurement
        # --- predict through the physical lag model
        prediction = self.x + (steady_state - self.x) * self.alpha
        p_pred = self.p * (1.0 - self.alpha) ** 2 + self.q

        # --- innovation and its covariance
        innovation = measurement - prediction
        s = p_pred + self.r
        sigma = math.sqrt(max(1e-12, s))
        detrended = innovation - self.baseline
        normalised = detrended / sigma

        # --- update
        k = p_pred / s
        self.x = prediction + k * innovation
        self.p = (1.0 - k) * p_pred

        # --- slow baseline adaptation, gated so real faults are not absorbed
        if abs(normalised) < BASELINE_GATE:
            self.baseline += BASELINE_RATE * detrended
        self.baseline = max(-BASELINE_LIMIT * sigma, min(BASELINE_LIMIT * sigma, self.baseline))

        return ChannelEstimate(
            channel=self.channel,
            measurement=measurement,
            prediction=prediction,
            estimate=self.x,
            steady_state=steady_state,
            innovation=innovation,
            innovation_sigma=sigma,
            normalised=normalised,
            nis=normalised ** 2,
            baseline_bias=self.baseline,
            baseline_sigma=self.baseline / sigma,
        )


class StateEstimator:
    """Bank of per-channel filters, plus per-cylinder EGT and CHT probes."""

    def __init__(self, dt: float | None = None, cylinders: int | None = None) -> None:
        dt = settings.sample_period_s if dt is None else dt
        self.dt = dt
        self.filters = {
            name: ChannelFilter(name, m['tau'], m['q'], m['r'], dt)
            for name, m in CHANNEL_MODELS.items()
        }
        n = settings.engine.cylinders if cylinders is None else cylinders
        self.cyl_egt = [ChannelFilter(f'egt_cyl{i + 1}', 2.5, 20.0, 25.0, dt) for i in range(n)]
        self.cyl_cht = [ChannelFilter(f'cht_cyl{i + 1}', 20.0, 0.8, 4.0, dt) for i in range(n)]

    def step(self, measured: dict[str, float], steady_state: dict[str, float],
             cylinders: dict[str, list[float]] | None = None,
             cylinder_steady: dict[str, list[float]] | None = None) -> dict[str, ChannelEstimate]:
        out: dict[str, ChannelEstimate] = {}
        for name, filt in self.filters.items():
            if name not in measured or name not in steady_state:
                continue
            out[name] = filt.step(float(steady_state[name]), float(measured[name]))

        if cylinders and cylinder_steady:
            for i, filt in enumerate(self.cyl_egt):
                if i < len(cylinders.get('egt', [])) and i < len(cylinder_steady.get('egt', [])):
                    out[filt.channel] = filt.step(float(cylinder_steady['egt'][i]), float(cylinders['egt'][i]))
            for i, filt in enumerate(self.cyl_cht):
                if i < len(cylinders.get('cht', [])) and i < len(cylinder_steady.get('cht', [])):
                    out[filt.channel] = filt.step(float(cylinder_steady['cht'][i]), float(cylinders['cht'][i]))
        return out
