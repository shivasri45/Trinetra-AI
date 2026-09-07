"""Synthesis of a crank-order vibration waveform for a four-stroke engine.

Rather than emitting a single scalar "vibration" number, the simulator builds a
short accelerometer time block so that the diagnostic layer can perform genuine
order tracking. For a four-cylinder four-stroke engine:

* order 0.5  - one cylinder fires once every two revolutions, so a single
               dead cylinder puts energy at half the rotational order
* order 1.0  - rotating imbalance / bent shaft / prop imbalance
* order 2.0  - main firing frequency (cylinders / 2 events per revolution)
* order 4.0  - first harmonic of the firing frequency
* 500-950 Hz - broadband structural band excited by bearing and lubrication
               distress and by knock / rough combustion
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VibrationDrivers:
    """Physical excitation levels, in g, before structural noise is added."""
    imbalance: float = 0.10        # order 1.0
    firing: float = 0.55           # order 2.0
    firing_harmonic: float = 0.18  # order 4.0
    half_order: float = 0.04       # order 0.5 (per-cylinder asymmetry)
    broadband: float = 0.06        # 500-950 Hz band amplitude
    noise: float = 0.05            # white structural noise sigma
    am_depth: float = 0.0          # combustion instability amplitude modulation
    am_hz: float = 3.0
    jitter: float = 0.0            # cycle-to-cycle phase jitter, radians
    extra_orders: dict[float, float] = field(default_factory=dict)


def healthy_drivers(load_fraction: float) -> VibrationDrivers:
    """Nominal excitation for a healthy engine at a given load fraction.

    Single source of truth, used both by the simulator as its baseline and by the
    digital twin as its expected vibration level. Combustion forcing rises with
    load, which is why the firing order dominates a healthy spectrum.
    """
    load = max(0.0, min(1.2, load_fraction))
    return VibrationDrivers(
        imbalance=0.055 + 0.055 * load,
        firing=0.34 + 0.42 * load,
        firing_harmonic=0.11 + 0.13 * load,
        half_order=0.030 + 0.020 * load,
        broadband=0.045 + 0.030 * load,
        noise=0.045,
    )


def expected_rms(drivers: VibrationDrivers) -> float:
    """Overall RMS of the driver set.

    Independent sinusoids contribute A^2/2 each; the broadband band and the
    structural noise contribute their variances directly.
    """
    tonal = (drivers.imbalance ** 2 + drivers.firing ** 2
             + drivers.firing_harmonic ** 2 + drivers.half_order ** 2
             + sum(a ** 2 for a in drivers.extra_orders.values()))
    return float(np.sqrt(tonal / 2.0 + drivers.broadband ** 2 + drivers.noise ** 2))


class VibrationSynthesiser:
    """Builds one accelerometer block per simulation tick."""

    def __init__(self, sample_rate: int, block: int, rng: np.random.Generator) -> None:
        self.sample_rate = sample_rate
        self.block = block
        self.rng = rng
        self.phase = 0.0

    def block_seconds(self) -> float:
        return self.block / self.sample_rate

    def generate(self, rpm: float, drivers: VibrationDrivers) -> np.ndarray:
        f_rot = max(1.0, rpm) / 60.0
        t = np.arange(self.block) / self.sample_rate
        nyquist = self.sample_rate / 2.0
        signal = np.zeros(self.block)

        orders: dict[float, float] = {
            0.5: drivers.half_order,
            1.0: drivers.imbalance,
            2.0: drivers.firing,
            4.0: drivers.firing_harmonic,
        }
        orders.update(drivers.extra_orders)

        for order, amplitude in orders.items():
            freq = order * f_rot
            if amplitude <= 0.0 or freq >= nyquist * 0.95:
                continue
            jitter = drivers.jitter * self.rng.standard_normal() if drivers.jitter else 0.0
            phase = self.phase * order + jitter
            signal += amplitude * np.sin(2.0 * np.pi * freq * t + phase)

        if drivers.broadband > 0.0:
            # Narrow structural band, generated in the frequency domain.
            spectrum = np.zeros(self.block // 2 + 1, dtype=complex)
            freqs = np.fft.rfftfreq(self.block, 1.0 / self.sample_rate)
            band = (freqs >= 500.0) & (freqs <= 950.0)
            count = int(band.sum())
            if count:
                phases = self.rng.uniform(0.0, 2.0 * np.pi, count)
                spectrum[band] = np.exp(1j * phases)
                shaped = np.fft.irfft(spectrum, n=self.block)
                rms = float(np.sqrt(np.mean(shaped ** 2))) or 1.0
                signal += drivers.broadband * shaped / rms

        if drivers.am_depth > 0.0:
            signal *= 1.0 + drivers.am_depth * np.sin(2.0 * np.pi * drivers.am_hz * t)

        signal += self.rng.normal(0.0, drivers.noise, self.block)
        self.phase = float((self.phase + 2.0 * np.pi * f_rot * self.block_seconds()) % (2.0 * np.pi))
        return signal
