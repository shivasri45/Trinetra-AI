"""Binds one simulator/CAN frame to the twin state and the feature vector.

Both the live service and the offline dataset generator drive this class, which
guarantees that training features and inference features come from identical
code. The label is deliberately *not* handled here.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.digital_twin.twin import DigitalTwin
from backend.preprocessing.pipeline import FeatureBuilder


@dataclass
class SyncResult:
    twin: dict
    features: dict[str, float]


class TwinSynchroniser:
    def __init__(self) -> None:
        self.twin = DigitalTwin()
        self.features = FeatureBuilder()

    def reset(self) -> None:
        self.twin = DigitalTwin()
        self.features.reset()

    def process(self, telemetry: dict, cylinders: dict[str, list[float]],
                vibration: dict[str, float]) -> SyncResult:
        state = self.twin.state(telemetry, cylinders, vibration)
        features = self.features.transform(telemetry, state, vibration, cylinders)
        return SyncResult(twin=state, features=features)
