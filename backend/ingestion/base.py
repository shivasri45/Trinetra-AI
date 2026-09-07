"""Telemetry source abstraction.

The analytics stack never talks to the simulator directly. It consumes
:class:`IngestedFrame` objects from whatever source is configured, so replacing
the synthetic engine with a real CAN interface, a log replay or a test-rig feed
is a matter of adding another implementation of :class:`TelemetrySource`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class IngestedFrame:
    """One acquisition cycle, as delivered to the analytics stack."""
    telemetry: dict
    cylinders: dict[str, list[float]]
    vibration: dict[str, float]
    source: str
    waveform: np.ndarray | None = None
    raw_frames: list[dict] = field(default_factory=list)
    transport: dict = field(default_factory=dict)
    # Simulator-only ground truth. Present for demonstration and offline scoring,
    # never passed to the twin, the feature builder or the models.
    truth: dict | None = None


@runtime_checkable
class TelemetrySource(Protocol):
    name: str

    def read(self) -> IngestedFrame:
        """Acquire the next frame."""
        ...

    def close(self) -> None:
        ...
