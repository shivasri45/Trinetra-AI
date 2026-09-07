"""Concrete telemetry sources: direct in-process and CAN bus.

``DirectSource`` hands the simulator's frame straight to the analytics stack. It
is the fastest path and is what the unit tests use.

``CanBusSource`` puts the full interface control document in the loop. Every
sample is encoded into J1939-style frames, transmitted on a SocketCAN-compatible
bus, received back, and reassembled by a decoder that has no access to the
original floating point values. That means the pipeline actually experiences bus
quantisation, signal ranges and frame assembly exactly as it would against a real
ECU, and swapping ``interface='virtual'`` for ``interface='socketcan'`` with
``channel='can0'`` is the only change needed to talk to hardware.
"""
from __future__ import annotations

import time

from backend.config.settings import settings
from backend.ingestion import can_frames
from backend.ingestion.base import IngestedFrame
from backend.ingestion.can_frames import MESSAGES, MESSAGES_BY_ID
from backend.simulator.engine import EngineSimulator

try:
    import can  # type: ignore
    HAS_PYTHON_CAN = True
except Exception:  # pragma: no cover - python-can is optional
    can = None
    HAS_PYTHON_CAN = False


class DirectSource:
    """In-process source; no serialisation, used for tests and fast replay."""

    name = 'direct'

    def __init__(self, simulator: EngineSimulator | None = None) -> None:
        self.simulator = simulator or EngineSimulator()

    def read(self) -> IngestedFrame:
        frame = self.simulator.step()
        return IngestedFrame(
            telemetry=frame.telemetry.dict(),
            cylinders=frame.cylinders,
            vibration=frame.vibration_features,
            waveform=frame.waveform,
            source=self.name,
            transport={'mode': 'direct', 'note': 'in-process, no bus encoding'},
            truth=dict(frame.truth),
        )

    def close(self) -> None:
        return None


class _LoopbackBus:
    """Minimal stand-in used when python-can is unavailable.

    The encode/decode path and the ICD are still fully exercised; only the
    transport is simplified.
    """

    def __init__(self) -> None:
        self._queue: list[tuple[int, bytes]] = []

    def send(self, arbitration_id: int, data: bytes) -> None:
        self._queue.append((arbitration_id, data))

    def drain(self) -> list[tuple[int, bytes]]:
        out, self._queue = self._queue, []
        return out


class CanBusSource:
    """Encodes each sample onto a CAN bus and decodes it back off the bus."""

    name = 'can'

    def __init__(self, simulator: EngineSimulator | None = None,
                 channel: str | None = None, interface: str | None = None,
                 keep_frames: int = 44) -> None:
        self.simulator = simulator or EngineSimulator()
        self.channel = channel or settings.can_channel
        self.interface = interface or settings.can_interface
        self.keep_frames = keep_frames
        self.bitrate = 250_000
        self._tx = None
        self._rx = None
        self._loopback: _LoopbackBus | None = None
        self._open()

    # ---------------------------------------------------------------- transport
    def _open(self) -> None:
        if HAS_PYTHON_CAN:
            try:
                self._tx = can.Bus(channel=self.channel, interface=self.interface,
                                   bitrate=self.bitrate)
                self._rx = can.Bus(channel=self.channel, interface=self.interface,
                                   bitrate=self.bitrate)
                self.backend = f'python-can:{self.interface}'
                return
            except Exception as exc:  # pragma: no cover - depends on host config
                self.open_error = str(exc)
        self._loopback = _LoopbackBus()
        self.backend = 'internal-loopback'

    def _send(self, arbitration_id: int, data: bytes) -> None:
        if self._tx is not None:
            self._tx.send(can.Message(arbitration_id=arbitration_id, data=data,
                                      is_extended_id=True), timeout=0.5)
        else:
            self._loopback.send(arbitration_id, data)

    def _receive(self, expected: int) -> list[tuple[int, bytes]]:
        if self._rx is None:
            return self._loopback.drain()
        received: list[tuple[int, bytes]] = []
        deadline = time.monotonic() + 0.5
        while len(received) < expected and time.monotonic() < deadline:
            message = self._rx.recv(timeout=0.05)
            if message is None:
                continue
            received.append((message.arbitration_id, bytes(message.data)))
        return received

    # -------------------------------------------------------------------- read
    def read(self) -> IngestedFrame:
        frame = self.simulator.step()
        telemetry = frame.telemetry.dict()
        signals = can_frames.build_signal_source(telemetry, frame.cylinders,
                                                 frame.vibration_features)

        sent: list[dict] = []
        for message in MESSAGES:
            payload = can_frames.encode(message, signals)
            self._send(message.can_id, payload)
            sent.append({'message': message.name, 'can_id': f'0x{message.can_id:08X}',
                         'pgn': f'0x{message.pgn:04X}', 'data': payload.hex(' ').upper()})

        decoded: dict[str, float] = {}
        seen: list[str] = []
        for arbitration_id, payload in self._receive(len(MESSAGES)):
            message = MESSAGES_BY_ID.get(arbitration_id)
            if message is None:
                continue
            decoded.update(can_frames.decode(message, payload))
            seen.append(message.name)

        rebuilt = self._reassemble(decoded, telemetry)
        return IngestedFrame(
            telemetry=rebuilt['telemetry'],
            cylinders=rebuilt['cylinders'],
            vibration=rebuilt['vibration'],
            waveform=frame.waveform,
            source=self.name,
            raw_frames=sent[-self.keep_frames:],
            transport={
                'mode': 'can',
                'backend': self.backend,
                'interface': self.interface,
                'channel': self.channel,
                'bitrate': self.bitrate,
                'frames_per_sample': len(MESSAGES),
                'bytes_per_sample': len(MESSAGES) * 8,
                'messages_received': seen,
                'waveform_note': 'accelerometer stream processed on the edge node; '
                                 'only condition indicators are transmitted',
            },
            truth=dict(frame.truth),
        )

    @staticmethod
    def _reassemble(decoded: dict[str, float], reference: dict) -> dict:
        """Rebuild the analytics-facing structures purely from decoded signals."""
        telemetry: dict = {'timestamp': reference['timestamp']}
        cylinders: dict[str, list[float]] = {'egt': [], 'cht': []}
        vibration: dict[str, float] = {}

        for name, value in decoded.items():
            if name in can_frames.CYLINDER_SIGNALS:
                channel = 'egt' if name.startswith('egt_cyl') else 'cht'
                index = int(name[-1]) - 1
                while len(cylinders[channel]) <= index:
                    cylinders[channel].append(0.0)
                cylinders[channel][index] = round(value, 2)
            elif name.startswith('vib_'):
                vibration[name[4:]] = round(value, 5)
            else:
                telemetry[name] = round(value, 4)

        # Integer signals that the bus carries as unsigned need their identity back.
        for key in ('hottest_cylinder', 'coldest_cylinder'):
            if cylinders['egt']:
                pick = max if key.startswith('hot') else min
                telemetry[key] = int(pick(range(len(cylinders['egt'])),
                                          key=lambda i: cylinders['egt'][i])) + 1
        vibration.setdefault('overall_rms', telemetry.get('vibration', 0.0))
        vibration['misfire_index'] = _ratio(vibration.get('order_0p5'), vibration.get('order_2p0'))
        vibration['imbalance_index'] = _ratio(vibration.get('order_1p0'), vibration.get('order_2p0'))
        vibration['roughness_index'] = _ratio(vibration.get('broadband_rms'), vibration.get('order_2p0'))
        return {'telemetry': telemetry, 'cylinders': cylinders, 'vibration': vibration}

    def close(self) -> None:
        for bus in (self._tx, self._rx):
            try:
                if bus is not None:
                    bus.shutdown()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self._tx = self._rx = None


def _ratio(numerator: float | None, denominator: float | None) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator or 0.0) / float(denominator), 4)


def build_source(mode: str, simulator: EngineSimulator) -> DirectSource | CanBusSource:
    if mode == 'can':
        return CanBusSource(simulator)
    return DirectSource(simulator)
