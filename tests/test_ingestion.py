"""CAN framing, round-trip fidelity and the ingestion source abstraction."""
from __future__ import annotations

import pytest

from backend.ingestion import can_frames
from backend.ingestion.base import IngestedFrame, TelemetrySource
from backend.ingestion.can_frames import MESSAGES, MESSAGES_BY_ID, Message, Signal
from backend.ingestion.sources import CanBusSource, DirectSource, build_source
from backend.simulator.engine import EngineSimulator

# Per-signal tolerance, driven by each signal's declared bus resolution.
TOLERANCE = {
    'rpm': 0.13, 'cht': 0.05, 'egt': 0.05, 'oil_temperature': 0.05,
    'ambient_temperature': 0.05, 'oil_pressure': 0.02, 'battery_voltage': 0.002,
    'alternator_output': 0.02, 'fuel_flow': 0.02, 'air_fuel_ratio': 0.002,
    'power_kw': 0.02, 'manifold_pressure': 0.02, 'injection_timing': 0.02,
    'ambient_pressure': 0.02, 'density_ratio': 0.0002, 'pressure_altitude': 0.6,
    'airspeed_ratio': 0.0002, 'engine_hours': 0.06, 'egt_spread': 0.11,
    'cht_spread': 0.11, 'throttle': 0.5, 'engine_load': 0.5, 'vibration': 0.0002,
}


# ------------------------------------------------------------------- framing
def test_can_identifiers_are_extended_and_well_formed():
    for message in MESSAGES:
        assert 0 < message.can_id <= 0x1FFFFFFF, message.name
        assert (message.can_id >> 8) & 0xFFFF == message.pgn
        assert message.can_id & 0xFF == can_frames.SOURCE_ADDRESS


def test_message_identifiers_and_names_are_unique():
    assert len(MESSAGES_BY_ID) == len(MESSAGES)
    names = [m.name for m in MESSAGES]
    assert len(set(names)) == len(names)


def test_no_message_exceeds_the_classic_can_payload():
    for message in MESSAGES:
        assert sum(signal.length for signal in message.signals) <= 8, message.name


def test_signal_names_are_globally_unique():
    names = [signal.name for message in MESSAGES for signal in message.signals]
    assert len(set(names)) == len(names)


def test_encode_produces_exactly_eight_bytes():
    for message in MESSAGES:
        assert len(can_frames.encode(message, {})) == 8


def test_encode_decode_round_trip_on_a_synthetic_message():
    message = Message('TEST', 0xFF00, (
        Signal('temperature', 2, 0.03125, -273.0),
        Signal('pressure', 2, 0.01),
    ))
    payload = can_frames.encode(message, {'temperature': 187.5, 'pressure': 54.32})
    decoded = can_frames.decode(message, payload)
    assert decoded['temperature'] == pytest.approx(187.5, abs=0.03125)
    assert decoded['pressure'] == pytest.approx(54.32, abs=0.01)


def test_out_of_range_values_saturate_rather_than_wrapping():
    message = Message('TEST', 0xFF01, (Signal('value', 1, 1.0),))
    assert can_frames.decode(message, can_frames.encode(message, {'value': 900.0}))['value'] == 255.0
    assert can_frames.decode(message, can_frames.encode(message, {'value': -50.0}))['value'] == 0.0


def test_raw_accelerometer_stream_is_not_on_the_bus():
    """Bandwidth argument: only edge-computed indicators are transmitted."""
    names = {signal.name for message in MESSAGES for signal in message.signals}
    assert not any('waveform' in name or 'samples' in name for name in names)
    assert 'vib_order_2p0' in names and 'vib_crest_factor' in names
    assert len(MESSAGES) * 8 < 200, 'per-sample bus load should stay small'


def test_interface_control_document_is_complete():
    rows = can_frames.interface_control_rows()
    assert len(rows) == len(MESSAGES)
    for row in rows:
        assert row['can_id'].startswith('0x') and row['pgn'].startswith('0x')
        assert row['signals']
        offsets = [s['start_byte'] for s in row['signals']]
        assert offsets == sorted(offsets), 'signal layout must be ordered'
        for signal in row['signals']:
            assert signal['resolution'] > 0


# -------------------------------------------------------------------- sources
def test_both_sources_satisfy_the_protocol():
    assert isinstance(DirectSource(EngineSimulator(1)), TelemetrySource)
    source = CanBusSource(EngineSimulator(1))
    try:
        assert isinstance(source, TelemetrySource)
    finally:
        source.close()


def test_build_source_selects_the_transport():
    assert isinstance(build_source('direct', EngineSimulator(1)), DirectSource)
    source = build_source('can', EngineSimulator(1))
    try:
        assert isinstance(source, CanBusSource)
    finally:
        source.close()


def test_direct_source_returns_a_populated_frame():
    frame = DirectSource(EngineSimulator(1)).read()
    assert isinstance(frame, IngestedFrame)
    assert frame.source == 'direct'
    assert frame.telemetry['rpm'] > 0
    assert frame.waveform is not None
    assert frame.cylinders['egt'] and frame.cylinders['cht']


def test_can_source_reassembles_the_frame_faithfully():
    """The decoder never sees the original floats, only the bus payloads."""
    direct = DirectSource(EngineSimulator(5))
    bus = CanBusSource(EngineSimulator(5))
    try:
        for _ in range(50):
            plain = direct.read()
            decoded = bus.read()
        for signal, tolerance in TOLERANCE.items():
            if signal in plain.telemetry and signal in decoded.telemetry:
                assert decoded.telemetry[signal] == pytest.approx(
                    plain.telemetry[signal], abs=tolerance), signal
        for channel in ('egt', 'cht'):
            for expected, actual in zip(plain.cylinders[channel], decoded.cylinders[channel]):
                assert actual == pytest.approx(expected, abs=0.05)
    finally:
        bus.close()


def test_can_source_reports_transport_detail():
    bus = CanBusSource(EngineSimulator(2))
    try:
        frame = bus.read()
        transport = frame.transport
        assert transport['mode'] == 'can'
        assert transport['frames_per_sample'] == len(MESSAGES)
        assert transport['bytes_per_sample'] == len(MESSAGES) * 8
        assert transport['bitrate'] == 250_000
        assert len(frame.raw_frames) == len(MESSAGES)
        for raw in frame.raw_frames:
            assert raw['can_id'].startswith('0x')
            assert len(raw['data'].split()) == 8
    finally:
        bus.close()


def test_can_source_recovers_derived_vibration_indices():
    bus = CanBusSource(EngineSimulator(6))
    try:
        for _ in range(20):
            frame = bus.read()
        vibration = frame.vibration
        for key in ('overall_rms', 'order_0p5', 'order_2p0', 'misfire_index',
                    'imbalance_index', 'roughness_index'):
            assert key in vibration, key
        expected = vibration['order_0p5'] / vibration['order_2p0']
        assert vibration['misfire_index'] == pytest.approx(expected, abs=1e-3)
    finally:
        bus.close()


def test_can_transport_preserves_diagnostic_capability():
    """A misfire must remain visible after passing through the bus."""
    bus = CanBusSource(EngineSimulator(7))
    try:
        bus.simulator.configure_mission('normal_cruise')
        for _ in range(90):
            baseline = bus.read()
        bus.simulator.inject_fault('misfire')
        for _ in range(60):
            frame = bus.read()
        assert frame.vibration['misfire_index'] > 5 * baseline.vibration['misfire_index']
        assert frame.telemetry['egt_spread'] > 200
    finally:
        bus.close()


def test_ground_truth_is_carried_separately_from_telemetry():
    frame = DirectSource(EngineSimulator(1)).read()
    assert frame.truth is not None and 'fault' in frame.truth
    assert 'fault' not in frame.telemetry
