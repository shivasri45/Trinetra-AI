# Interface Control Document - Engine Data Bus

Generated from `backend/ingestion/can_frames.py` by `python -m tools.generate_icd_doc`.
The machine-readable form is served at `GET /interface-control`. If this file and
the code disagree, the code is correct.

## Bus

| Property | Value |
| --- | --- |
| Physical layer | CAN 2.0B, extended 29-bit identifiers |
| Bit rate | 250 kbit/s |
| Addressing | SAE J1939 style: priority 3, PGN, source address |
| Source address | `0x00` (engine ECU / FADEC) |
| Frames per sample | 11 |
| Payload per sample | 88 bytes |
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

The raw accelerometer stream. At 2048 Hz it would need roughly
66 kB/s, which is not a reasonable load for a shared 250 kbit/s engine
bus. Sampling, windowed FFT and RPM-synchronous order tracking run on the edge
node beside the engine, and only the derived condition indicators are
transmitted - three frames per sample.

## Messages


### EEC1 - PGN 0xF004

Electronic engine controller: speed, throttle, load, injection advance, MAP.

CAN ID `0x0CF00400`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `rpm` | 0.125 | 0 | 0 to 8192 |
| 2 | 1 | `throttle` | 0.4 | 0 | 0 to 102 |
| 3 | 1 | `engine_load` | 0.4 | 0 | 0 to 102 |
| 4 | 2 | `injection_timing` | 0.01 | -100 | -100 to 555.4 |
| 6 | 2 | `manifold_pressure` | 0.01 | 0 | 0 to 655.4 |

### ET1 - PGN 0xFEEE

Engine temperatures.

CAN ID `0x0CFEEE00`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `cht` | 0.03125 | -273 | -273 to 1775 |
| 2 | 2 | `oil_temperature` | 0.03125 | -273 | -273 to 1775 |
| 4 | 2 | `egt` | 0.03125 | -273 | -273 to 1775 |
| 6 | 2 | `ambient_temperature` | 0.03125 | -273 | -273 to 1775 |

### EFLP1 - PGN 0xFEEF

Engine fluid pressure and electrical bus.

CAN ID `0x0CFEEF00`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `oil_pressure` | 0.01 | 0 | 0 to 655.4 |
| 2 | 2 | `battery_voltage` | 0.001 | 0 | 0 to 65.53 |
| 4 | 2 | `alternator_output` | 0.01 | 0 | 0 to 655.4 |

### LFE - PGN 0xFEF2

Fuel economy and delivered shaft power.

CAN ID `0x0CFEF200`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `fuel_flow` | 0.01 | 0 | 0 to 655.4 |
| 2 | 2 | `air_fuel_ratio` | 0.001 | 0 | 0 to 65.53 |
| 4 | 2 | `power_kw` | 0.01 | 0 | 0 to 655.4 |

### AMB - PGN 0xFEF5

Atmospheric conditions and cooling airflow reference.

CAN ID `0x0CFEF500`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `ambient_pressure` | 0.01 | 0 | 0 to 655.4 |
| 2 | 2 | `density_ratio` | 0.0001 | 0 | 0 to 6.554 |
| 4 | 2 | `pressure_altitude` | 0.5 | -1000 | -1000 to 3.177e+04 |
| 6 | 2 | `airspeed_ratio` | 0.0001 | 0 | 0 to 6.554 |

### EGT_CYL - PGN 0xFF10

Proprietary: per-cylinder exhaust gas temperature.

CAN ID `0x0CFF1000`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `egt_cyl1` | 0.03125 | -273 | -273 to 1775 |
| 2 | 2 | `egt_cyl2` | 0.03125 | -273 | -273 to 1775 |
| 4 | 2 | `egt_cyl3` | 0.03125 | -273 | -273 to 1775 |
| 6 | 2 | `egt_cyl4` | 0.03125 | -273 | -273 to 1775 |

### CHT_CYL - PGN 0xFF11

Proprietary: per-cylinder head temperature.

CAN ID `0x0CFF1100`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `cht_cyl1` | 0.03125 | -273 | -273 to 1775 |
| 2 | 2 | `cht_cyl2` | 0.03125 | -273 | -273 to 1775 |
| 4 | 2 | `cht_cyl3` | 0.03125 | -273 | -273 to 1775 |
| 6 | 2 | `cht_cyl4` | 0.03125 | -273 | -273 to 1775 |

### VIB_LEVEL - PGN 0xFF12

Proprietary: edge-computed overall vibration condition indicators.

CAN ID `0x0CFF1200`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `vibration` | 0.0001 | 0 | 0 to 6.554 |
| 2 | 2 | `vib_crest_factor` | 0.001 | 0 | 0 to 65.53 |
| 4 | 2 | `vib_spectral_kurtosis` | 0.01 | 0 | 0 to 655.4 |
| 6 | 2 | `vib_broadband_rms` | 0.0001 | 0 | 0 to 6.554 |

### VIB_ORDERS_A - PGN 0xFF13

Proprietary: edge-computed crank-order amplitudes, orders 0.5 to 2.

CAN ID `0x0CFF1300`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `vib_order_0p5` | 0.0001 | 0 | 0 to 6.554 |
| 2 | 2 | `vib_order_1p0` | 0.0001 | 0 | 0 to 6.554 |
| 4 | 2 | `vib_order_1p5` | 0.0001 | 0 | 0 to 6.554 |
| 6 | 2 | `vib_order_2p0` | 0.0001 | 0 | 0 to 6.554 |

### VIB_ORDERS_B - PGN 0xFF14

Proprietary: edge-computed crank-order amplitudes, orders 3 and 4.

CAN ID `0x0CFF1400`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `vib_order_3p0` | 0.0001 | 0 | 0 to 6.554 |
| 2 | 2 | `vib_order_4p0` | 0.0001 | 0 | 0 to 6.554 |
| 4 | 2 | `vib_rotational_hz` | 0.01 | 0 | 0 to 655.4 |
| 6 | 2 | `vib_firing_hz` | 0.01 | 0 | 0 to 655.4 |

### ENG_HOURS - PGN 0xFF15

Proprietary: accumulated hours and per-cylinder temperature spread.

CAN ID `0x0CFF1500`, extended identifier, 8 data bytes.

| Byte | Length | Signal | Resolution | Offset | Representable range |
| --- | --- | --- | --- | --- | --- |
| 0 | 4 | `engine_hours` | 0.05 | 0 | 0 to 2.147e+08 |
| 4 | 2 | `egt_spread` | 0.1 | 0 | 0 to 6554 |
| 6 | 2 | `cht_spread` | 0.1 | 0 | 0 to 6554 |

## Moving from the virtual bus to hardware

`backend/ingestion/sources.py` uses `python-can`, so the transport is a settings
change and nothing else:

```
TRINETRA_CAN_INTERFACE=virtual   TRINETRA_CAN_CHANNEL=trinetra   # demonstration
TRINETRA_CAN_INTERFACE=socketcan TRINETRA_CAN_CHANNEL=can0       # Linux hardware
```

No analytics code changes. The decoder already reconstructs every analytics input
from bus payloads alone, so the pipeline experiences real quantisation, signal
ranges and frame assembly rather than in-process floating point values.
