"""Order tracking and spectral condition indicators for engine vibration.

The analysis is deliberately RPM-synchronous (order based rather than fixed
frequency) so that the resulting features stay comparable across the engine's
speed range, which is what makes them usable as machine-learning inputs.
"""
from __future__ import annotations

import numpy as np

TRACKED_ORDERS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
BROADBAND_HZ: tuple[float, float] = (500.0, 950.0)


def _amplitude_at(freqs: np.ndarray, magnitude: np.ndarray, target_hz: float, half_width_hz: float) -> float:
    """Peak amplitude in a narrow window, tolerating small speed fluctuations."""
    if target_hz <= 0.0 or target_hz >= freqs[-1]:
        return 0.0
    window = (freqs >= target_hz - half_width_hz) & (freqs <= target_hz + half_width_hz)
    if not window.any():
        return 0.0
    return float(magnitude[window].max())


def analyse(waveform: np.ndarray, rpm: float, sample_rate: int) -> dict[str, float]:
    """Return overall level plus per-order and band condition indicators."""
    n = len(waveform)
    centred = waveform - float(np.mean(waveform))
    overall_rms = float(np.sqrt(np.mean(centred ** 2)))
    peak = float(np.max(np.abs(centred))) if n else 0.0
    crest = peak / overall_rms if overall_rms > 1e-9 else 0.0

    window = np.hanning(n)
    coherent_gain = float(np.mean(window))
    spectrum = np.fft.rfft(centred * window)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    # Single-sided amplitude spectrum corrected for window coherent gain.
    magnitude = np.abs(spectrum) * 2.0 / (n * coherent_gain)

    f_rot = max(1.0, rpm) / 60.0
    resolution = sample_rate / n
    half_width = max(1.5 * resolution, 0.03 * f_rot)

    features: dict[str, float] = {
        'overall_rms': round(overall_rms, 4),
        'peak': round(peak, 4),
        'crest_factor': round(crest, 3),
        'rotational_hz': round(f_rot, 2),
        'firing_hz': round(2.0 * f_rot, 2),
    }
    for order in TRACKED_ORDERS:
        amp = _amplitude_at(freqs, magnitude, order * f_rot, half_width)
        features[f'order_{str(order).replace(".", "p")}'] = round(amp, 4)

    band = (freqs >= BROADBAND_HZ[0]) & (freqs <= BROADBAND_HZ[1])
    band_rms = float(np.sqrt(np.sum(magnitude[band] ** 2) / 2.0)) if band.any() else 0.0
    features['broadband_rms'] = round(band_rms, 4)

    # Spectral kurtosis flags impulsive content typical of bearing distress.
    if magnitude.size > 4:
        power = magnitude ** 2
        mean_p = float(np.mean(power))
        std_p = float(np.std(power))
        features['spectral_kurtosis'] = round(float(np.mean(((power - mean_p) / std_p) ** 4)), 3) if std_p > 1e-12 else 0.0
    else:
        features['spectral_kurtosis'] = 0.0

    firing = features['order_2p0']
    half = features['order_0p5']
    # Half-order to firing-order ratio is the classic single-cylinder misfire index.
    features['misfire_index'] = round(half / firing, 4) if firing > 1e-6 else 0.0
    features['imbalance_index'] = round(features['order_1p0'] / firing, 4) if firing > 1e-6 else 0.0
    features['roughness_index'] = round(band_rms / firing, 4) if firing > 1e-6 else 0.0
    return features


def spectrum_for_display(waveform: np.ndarray, sample_rate: int, max_hz: float = 900.0,
                         bins: int = 96) -> list[dict[str, float]]:
    """Down-sampled amplitude spectrum for the dashboard chart."""
    n = len(waveform)
    if n == 0:
        return []
    centred = waveform - float(np.mean(waveform))
    window = np.hanning(n)
    magnitude = np.abs(np.fft.rfft(centred * window)) * 2.0 / (n * float(np.mean(window)))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    keep = freqs <= max_hz
    freqs, magnitude = freqs[keep], magnitude[keep]
    if freqs.size == 0:
        return []
    edges = np.linspace(0.0, float(freqs[-1]), bins + 1)
    out: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (freqs >= lo) & (freqs < hi)
        amp = float(magnitude[sel].max()) if sel.any() else 0.0
        out.append({'hz': round(float((lo + hi) / 2.0), 1), 'amplitude': round(amp, 4)})
    return out
