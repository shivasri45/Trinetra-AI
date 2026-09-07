# Deployment Roadmap

Path from the current synthetic prototype to a fielded engine health monitoring
capability. Written to be honest about what exists today and what is required
before anything touches a real aircraft.

---

## Where the prototype stands

| Capability | Status |
| --- | --- |
| Thermodynamic engine model | Implemented from first principles, dimensionally realistic, **not calibrated to any real engine** |
| Kalman state estimation and health indices | Implemented, validated against transients |
| Fault detection and identification | Implemented, 0.986 held-out accuracy on synthetic data |
| RUL with confidence interval | Implemented, extrapolation from degradation trend |
| CAN / J1939-style ingestion | Implemented over `python-can`; virtual bus in the demonstration |
| Mission recording and replay | Implemented, SQLite |
| API-key auth, HMAC frame integrity, audit trail | Implemented |
| Dashboard: live, replay, model and interface views | Implemented |
| Real engine data | **None** |
| Airworthiness evidence | **None** |

---

## Stage 1 - Test rig correlation

**Objective:** replace assumed constants with measured ones.

The physics module carries a small number of calibration constants
(`CYCLE_DEVIATION`, `VE_MAX`, `VE_PEAK_RPM`, `COOLING_COEFF`,
`OIL_PUMP_PSI_REF`, `EXHAUST_HEAT_FRACTION`, `HEAD_HEAT_FRACTION`). Each is
currently chosen so the model reproduces published rating figures. Each needs to
come from rig measurement instead.

- Instrument a representative engine on a dynamometer with the channel set in the
  ICD, including per-cylinder EGT and CHT and a triaxial accelerometer.
- Map volumetric efficiency, BSFC and temperatures across the speed/load envelope
  and in an altitude chamber if available.
- Fit the calibration constants, then quantify model error per channel. That error
  becomes the process noise `Q` in the estimator, replacing today's engineering
  estimates.
- Measure real transducer accuracies and replace the assumed `R` values.

**Exit criteria:** steady-state model error within transducer accuracy across the
envelope; documented `Q` and `R` per channel.

## Stage 2 - Seeded fault campaign

**Objective:** replace synthetic fault physics with observed fault signatures.

- Run seeded faults on the rig: disabled injector, partially blocked injector,
  restricted cooling, controlled oil bypass, imbalanced propeller, drifted
  thermocouple, progressive intake fouling.
- Record full development where safe to do so, not just the end state, so the
  incipient stage is represented.
- Retrain on rig data and re-evaluate with the session-grouped split already used.
- Compare rig signatures against the simulator's and correct the fault models
  where they disagree.

**Exit criteria:** classifier trained on rig data; per-class detection latency and
false-alarm rate quantified on held-out rig sessions.

## Stage 3 - Edge node

**Objective:** move acquisition and signal processing onto hardware.

- Target an ARM-class SBC or an automotive-grade MCU with a CAN controller.
- Implement accelerometer sampling, windowed FFT and order tracking on the edge,
  matching `backend/signal_processing/spectral.py` bit-for-bit against a test
  vector set. The bus carries indicators only, as the ICD already specifies.
- Benchmark the inference budget. The RandomForest bundle is a few megabytes; if
  it does not fit, options are tree-count reduction, quantisation, or keeping
  classification at the ground station while the edge runs the statistical
  detector, which needs no model at all.
- Verify bus loading: 11 frames per sample at 1 Hz is negligible on a 250 kbit/s
  bus, leaving room to raise the sample rate.

**Exit criteria:** edge node producing ICD-conformant frames from a real engine;
spectral features matching the reference implementation within tolerance.

## Stage 4 - Ground station integration

**Objective:** operate inside a real GCS.

- Replace SQLite with PostgreSQL or TimescaleDB. This is a `Repository`-level
  change; nothing above it moves.
- Run the API behind TLS termination. HMAC frame signing covers integrity and
  authenticity but not confidentiality.
- Replace the static API key with the GCS identity provider, mapped to roles:
  operator (read), propulsion engineer (read plus replay), maintainer (read plus
  configuration). Fault injection must be disabled entirely outside test builds.
- Forward the audit trail to the station's central log.
- Define behaviour on telemetry loss, out-of-order frames and clock skew. The
  current estimator assumes a regular 1 Hz cadence.

**Exit criteria:** deployed in a GCS test environment, authenticated against the
station's identity provider, surviving link dropouts.

## Stage 5 - Flight test and fleet

**Objective:** evidence from operation, then scale.

- Shadow mode first: run alongside the existing threshold system with no
  operational authority, and compare every alert against maintenance findings.
- Track false alarms and missed detections against real teardown evidence. This is
  the only measurement that matters for trust.
- Validate RUL by comparing predictions against actual times to removal. Expect
  the first models to be poor; RUL needs the most real data of anything here.
- Fleet aggregation: per-airframe baselines, cross-fleet comparison to spot an
  outlier engine, and consolidated maintenance planning.
- Federated learning becomes worthwhile at this point: fleet operators can improve
  a shared model without pooling raw telemetry, which matters when the data is
  operationally sensitive.

**Exit criteria:** shadow-mode statistics good enough to justify advisory
authority; RUL error characterised against removals.

---

## Certification note

Nothing in this repository constitutes airworthiness evidence. A health monitoring
function that informs dispatch or maintenance decisions attracts certification
obligations, and machine-learned components are the difficult part: the standard
V-model assumes traceable requirements down to code, which a trained model does
not naturally provide.

The pragmatic route, and the one this architecture is arranged for:

- Keep the system **advisory**. Do not give it authority over engine control or
  dispatch release without the corresponding certification basis.
- Lean on the deterministic path for anything safety-related. The statistical
  detector is a Kalman filter over a physical model: reviewable, traceable and
  free of learned parameters. The physics prior in `diagnostics.py` is likewise
  inspectable rule logic. Both remain fully functional with no trained model
  present, which is why the fallback path exists and is tested.
- Treat the learned models as the layer that improves usefulness rather than the
  layer that provides assurance. Guidance for machine learning in aviation is
  still consolidating (EASA's learning assurance concept paper and the SAE
  G-34 / EUROCAE WG-114 work are the relevant threads); design so the learned
  part can be constrained or removed without losing the safety argument.
- Maintain model provenance from the start: training data version, split
  protocol, metrics and feature schema. `models/metrics.json` and
  `docs/model_card.md` are the beginning of that record, and `/model/metrics`
  makes the deployed model's provenance queryable at runtime.

---

## Engineering debt to clear before fielding

| Item | Why it matters |
| --- | --- |
| Single SQLite connection with `check_same_thread=False` | fine for one process, wrong for multi-worker serving |
| Wear acceleration factor of 1800 | a demonstration device; real deployment tracks real hours |
| Fault injection endpoints | must be absent from production builds, not merely access-controlled |
| Fixed 1 Hz cadence assumption | the estimator needs to handle variable and missing samples |
| No sensor range or validity checks on ingest | a failed transducer reading full-scale should be rejected before it reaches the filters |
| Model retraining is manual | needs a pipeline with versioning and automatic regression gates |
| No clock synchronisation strategy | mission replay and fleet comparison both depend on trustworthy timestamps |
