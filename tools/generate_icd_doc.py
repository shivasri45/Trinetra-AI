"""Generate docs/interface_control.md from the code that defines the bus.

Run after changing backend/ingestion/can_frames.py:

    python -m tools.generate_icd_doc

Generating the document rather than maintaining it by hand keeps the ICD from
drifting away from the frames actually transmitted.
"""
from __future__ import annotations

from pathlib import Path

from backend.ingestion import can_frames

OUTPUT = Path('docs/interface_control.md')

HEADER = """# Interface Control Document - Engine Data Bus

Generated from `backend/ingestion/can_frames.py` by `python -m tools.generate_icd_doc`.
The machine-readable form is served at `GET /interface-control`. If this file and
the code disagree, the code is correct.

## Bus

| Property | Value |
| --- | --- |
| Physical layer | CAN 2.0B, extended 29-bit identifiers |
| Bit rate | 250 kbit/s |
| Addressing | SAE J1939 style: priority {priority}, PGN, source address |
| Source address | `0x{source:02X}` (engine ECU / FADEC) |
| Frames per sample | {frames} |
| Payload per sample | {payload} bytes |
| Sample rate | 1 Hz |
| Byte order | little-endian within each signal |
| Encoding | `value = raw * resolution + offset` |

Standard PGNs are reused where one exists for the quantity (EEC1, ET1, EFL/P1,
LFE, AMB). Proprietary-B PGNs from `0xFF10` carry the per-cylinder probes and the
edge-computed vibration indicators.

Temperature signals follow the J1939 convention of 1/32 K per bit with a -273
offset, which is why their resolution reads as 0.03125.

## Transducer accuracies

These are the 1-sigma accuracies used as measurement noise `R` in the twin's
Kalman filters (`backend/digital_twin/estimator.py`). Declaring them honestly
matters: an over-optimistic sensor model makes ordinary instrument scatter and
normal wear look like faults, which is the usual source of nuisance alarms.

| Channel | 1-sigma accuracy |
| --- | --- |
| Engine speed | +/- 10 rpm |
| Cylinder head temperature | +/- 2 C (type-K thermocouple) |
| Exhaust gas temperature | +/- 5 C |
| Oil pressure | +/- 0.5 psi |
| Oil temperature | +/- 1 C |
| Fuel flow | +/- 2 % of reading |
| Manifold pressure | +/- 0.5 kPa |
| Air/fuel ratio | +/- 0.15 AFR |
| Injection advance | +/- 0.3 deg |
| Alternator current | +/- 0.3 A |
| Battery voltage | +/- 0.05 V |
| Vibration level | +/- 0.04 g |

## What is deliberately not on the bus

The raw accelerometer stream. At {vib_rate} Hz it would need roughly
{vib_load:.0f} kB/s, which is not a reasonable load for a shared 250 kbit/s engine
bus. Sampling, windowed FFT and RPM-synchronous order tracking run on the edge
node beside the engine, and only the derived condition indicators are
transmitted - three frames per sample.

## Messages

"""

FOOTER = """## Moving from the virtual bus to hardware

`backend/ingestion/sources.py` uses `python-can`, so the transport is a settings
change and nothing else:

```
TRINETRA_CAN_INTERFACE=virtual   TRINETRA_CAN_CHANNEL=trinetra   # demonstration
TRINETRA_CAN_INTERFACE=socketcan TRINETRA_CAN_CHANNEL=can0       # Linux hardware
```

No analytics code changes. The decoder already reconstructs every analytics input
from bus payloads alone, so the pipeline experiences real quantisation, signal
ranges and frame assembly rather than in-process floating point values.
"""


def build() -> str:
    from backend.config.settings import settings

    rows = can_frames.interface_control_rows()
    vib_load = settings.vib_sample_rate_hz * 4 / 1000.0  # 4 bytes per sample
    parts = [HEADER.format(
        priority=can_frames.DEFAULT_PRIORITY,
        source=can_frames.SOURCE_ADDRESS,
        frames=len(rows),
        payload=len(rows) * 8,
        vib_rate=settings.vib_sample_rate_hz,
        vib_load=vib_load * 8,
    )]

    for row in rows:
        parts.append(f"### {row['message']} - PGN {row['pgn']}\n")
        parts.append(f"{row['description']}.\n")
        parts.append(f"CAN ID `{row['can_id']}`, extended identifier, "
                     f"{row['length_bytes']} data bytes.\n")
        parts.append('| Byte | Length | Signal | Resolution | Offset | Representable range |')
        parts.append('| --- | --- | --- | --- | --- | --- |')
        for signal in row['signals']:
            span = signal['resolution'] * ((1 << (8 * signal['length_bytes'])) - 1)
            low = signal['offset']
            parts.append(
                f"| {signal['start_byte']} | {signal['length_bytes']} | `{signal['signal']}` "
                f"| {signal['resolution']:g} | {signal['offset']:g} "
                f"| {low:.4g} to {low + span:.4g} |")
        parts.append('')

    parts.append(FOOTER)
    return '\n'.join(parts)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(), encoding='utf-8')
    rows = can_frames.interface_control_rows()
    print(f'wrote {OUTPUT}: {len(rows)} messages, '
          f'{sum(len(r["signals"]) for r in rows)} signals')


if __name__ == '__main__':
    main()
