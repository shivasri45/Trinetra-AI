"""J1939-style frame definitions for the engine data bus.

Signals are packed into 8-byte payloads with an explicit resolution and offset
per signal, in the manner of SAE J1939 SPNs, using little-endian byte order.
Standard PGNs are reused where one exists for the quantity (EEC1, ET1, EFL/P1,
LFE, AMB) and proprietary-B PGNs carry everything specific to this installation.

An important architectural point is visible in the table: the raw accelerometer
stream never crosses the bus. Vibration is sampled and order-tracked on the edge
node next to the engine, and only the derived condition indicators are
transmitted. A 2 kHz waveform would need roughly 33 kB/s, which is not a
reasonable load for a shared 250 kbit/s engine bus, whereas the indicator set
fits in three frames per sample.
"""
from __future__ import annotations

from dataclasses import dataclass

# Bus arbitration: priority 3, source address 0x00 (engine ECU/FADEC).
DEFAULT_PRIORITY = 3
SOURCE_ADDRESS = 0x00
GLOBAL_DESTINATION = 0xFF


@dataclass(frozen=True)
class Signal:
    name: str
    length: int          # bytes
    resolution: float
    offset: float = 0.0
    signed: bool = False


@dataclass(frozen=True)
class Message:
    name: str
    pgn: int
    signals: tuple[Signal, ...]
    description: str = ''

    @property
    def can_id(self) -> int:
        """29-bit extended identifier: priority | PGN | source address."""
        return (DEFAULT_PRIORITY << 26) | (self.pgn << 8) | SOURCE_ADDRESS

    def layout(self) -> list[dict]:
        rows: list[dict] = []
        start = 0
        for signal in self.signals:
            rows.append({'signal': signal.name, 'start_byte': start, 'length_bytes': signal.length,
                         'resolution': signal.resolution, 'offset': signal.offset})
            start += signal.length
        return rows


# Temperature signals use the J1939 convention of 1/32 K per bit with a -273 K offset.
T = dict(length=2, resolution=0.03125, offset=-273.0)

MESSAGES: tuple[Message, ...] = (
    Message('EEC1', 0xF004, (
        Signal('rpm', 2, 0.125),
        Signal('throttle', 1, 0.4),
        Signal('engine_load', 1, 0.4),
        Signal('injection_timing', 2, 0.01, -100.0),
        Signal('manifold_pressure', 2, 0.01),
    ), 'Electronic engine controller: speed, throttle, load, injection advance, MAP'),

    Message('ET1', 0xFEEE, (
        Signal('cht', **T),
        Signal('oil_temperature', **T),
        Signal('egt', **T),
        Signal('ambient_temperature', **T),
    ), 'Engine temperatures'),

    Message('EFLP1', 0xFEEF, (
        Signal('oil_pressure', 2, 0.01),
        Signal('battery_voltage', 2, 0.001),
        Signal('alternator_output', 2, 0.01),
    ), 'Engine fluid pressure and electrical bus'),

    Message('LFE', 0xFEF2, (
        Signal('fuel_flow', 2, 0.01),
        Signal('air_fuel_ratio', 2, 0.001),
        Signal('power_kw', 2, 0.01),
    ), 'Fuel economy and delivered shaft power'),

    Message('AMB', 0xFEF5, (
        Signal('ambient_pressure', 2, 0.01),
        Signal('density_ratio', 2, 0.0001),
        Signal('pressure_altitude', 2, 0.5, -1000.0),
        Signal('airspeed_ratio', 2, 0.0001),
    ), 'Atmospheric conditions and cooling airflow reference'),

    Message('EGT_CYL', 0xFF10, (
        Signal('egt_cyl1', **T), Signal('egt_cyl2', **T),
        Signal('egt_cyl3', **T), Signal('egt_cyl4', **T),
    ), 'Proprietary: per-cylinder exhaust gas temperature'),

    Message('CHT_CYL', 0xFF11, (
        Signal('cht_cyl1', **T), Signal('cht_cyl2', **T),
        Signal('cht_cyl3', **T), Signal('cht_cyl4', **T),
    ), 'Proprietary: per-cylinder head temperature'),

    Message('VIB_LEVEL', 0xFF12, (
        Signal('vibration', 2, 0.0001),
        Signal('vib_crest_factor', 2, 0.001),
        Signal('vib_spectral_kurtosis', 2, 0.01),
        Signal('vib_broadband_rms', 2, 0.0001),
    ), 'Proprietary: edge-computed overall vibration condition indicators'),

    Message('VIB_ORDERS_A', 0xFF13, (
        Signal('vib_order_0p5', 2, 0.0001),
        Signal('vib_order_1p0', 2, 0.0001),
        Signal('vib_order_1p5', 2, 0.0001),
        Signal('vib_order_2p0', 2, 0.0001),
    ), 'Proprietary: edge-computed crank-order amplitudes, orders 0.5 to 2'),

    Message('VIB_ORDERS_B', 0xFF14, (
        Signal('vib_order_3p0', 2, 0.0001),
        Signal('vib_order_4p0', 2, 0.0001),
        Signal('vib_rotational_hz', 2, 0.01),
        Signal('vib_firing_hz', 2, 0.01),
    ), 'Proprietary: edge-computed crank-order amplitudes, orders 3 and 4'),

    Message('ENG_HOURS', 0xFF15, (
        Signal('engine_hours', 4, 0.05),
        Signal('egt_spread', 2, 0.1),
        Signal('cht_spread', 2, 0.1),
    ), 'Proprietary: accumulated hours and per-cylinder temperature spread'),
)

MESSAGES_BY_ID: dict[int, Message] = {m.can_id: m for m in MESSAGES}

# Signals that belong to the vibration indicator set rather than the telemetry frame.
VIBRATION_SIGNALS: frozenset[str] = frozenset(
    signal.name[4:] for message in MESSAGES for signal in message.signals
    if signal.name.startswith('vib_'))
CYLINDER_SIGNALS: frozenset[str] = frozenset(
    signal.name for message in MESSAGES for signal in message.signals
    if signal.name.startswith(('egt_cyl', 'cht_cyl')))


def _clamp_raw(raw: int, length: int) -> int:
    return max(0, min((1 << (8 * length)) - 1, raw))


def encode(message: Message, values: dict[str, float]) -> bytes:
    payload = bytearray()
    for signal in message.signals:
        value = float(values.get(signal.name, 0.0))
        raw = int(round((value - signal.offset) / signal.resolution))
        payload += _clamp_raw(raw, signal.length).to_bytes(signal.length, 'little')
    return bytes(payload).ljust(8, b'\x00')[:8]


def decode(message: Message, payload: bytes) -> dict[str, float]:
    values: dict[str, float] = {}
    index = 0
    for signal in message.signals:
        chunk = payload[index:index + signal.length]
        if len(chunk) < signal.length:
            break
        raw = int.from_bytes(chunk, 'little')
        values[signal.name] = raw * signal.resolution + signal.offset
        index += signal.length
    return values


def build_signal_source(telemetry: dict, cylinders: dict[str, list[float]],
                        vibration: dict[str, float]) -> dict[str, float]:
    """Flatten the engine state into the flat signal namespace used by the bus."""
    values = dict(telemetry)
    for channel in ('egt', 'cht'):
        for i, value in enumerate(cylinders.get(channel, []), start=1):
            values[f'{channel}_cyl{i}'] = value
    for name, value in vibration.items():
        values[f'vib_{name}'] = value
    return values


def interface_control_rows() -> list[dict]:
    """Machine-readable ICD, exposed through the API and used to build the docs."""
    return [{
        'message': m.name,
        'pgn': f'0x{m.pgn:04X}',
        'can_id': f'0x{m.can_id:08X}',
        'extended': True,
        'length_bytes': 8,
        'description': m.description,
        'signals': m.layout(),
    } for m in MESSAGES]
