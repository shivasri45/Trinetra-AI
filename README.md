# Trinetra AI - Aero Piston Engine Digital Twin

Real-time digital twin, health monitoring, fault prediction and mission
reliability tooling for an aero piston engine of the class used in MALE UAVs.

A thermodynamic engine model is solved from first principles and synchronised to
live telemetry by a per-channel Kalman filter. The difference between the two
drives subsystem health indices, fault detection and identification, remaining
useful life with a confidence interval, and dispatch and maintenance advice.

> **Every value in this project is synthetic.** The model is not calibrated
> against any real engine and nothing here is certified for flight, maintenance or
> airworthiness decisions. See `docs/deployment_roadmap.md` for what would be
> required.

---

## Quick start

Backend (Python 3.11+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train the diagnostic models (about 4 minutes; models/ ships empty)
python -m backend.simulator.generate_dataset
python -m backend.ml.train

python -m uvicorn backend.main:app --reload
```

Frontend (Node 18+), in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` and press **Start engine**. API documentation is at
`http://localhost:8000/docs`.

Without a trained model the system still runs: diagnostics fall back to the
transparent physics rules and report `model_backend: physics_prior_fallback`.

---

## Demonstration script

1. **Start the engine.** Health settles near 99. All five mission profiles read
   NORMAL.
2. **Select *Rapid Throttle Transition*.** Health stays normal through repeated
   full-power and idle cycles. Open the *Twin synchronisation* panel: innovations
   stay near the noise floor while the raw expected-versus-measured gap swings by
   over 100 sigma. This is the false-alarm problem that threshold monitoring has.
3. **Inject *Misfire*.** Within ~3 s: cylinder 3's EGT collapses, the vibration
   spectrum grows a half-crank-order line, the combustion index drops, and the
   diagnosis names the fault with its evidence attributed to specific channels.
4. **Inject *Sensor Drift*.** CHT climbs 46 C while oil temperature and EGT do
   not. Threshold monitoring would report overheating. The twin reports a suspect
   instrument, and health stays high because the engine is fine.
5. **Compare with *Overheating*.** The same CHT rise, but oil temperature follows.
   The thermal chain stays mutually consistent, so this one is a real thermal
   event.
6. **Watch the RUL panel.** Per-subsystem remaining life with 95 % confidence
   bands, the limiting subsystem named, and a specific maintenance action.
7. **Switch ingestion to *CAN bus*.** Identical analytics, now fed by decoded
   J1939-style frames. The *Models and interfaces* tab shows live frames with
   arbitration IDs and payload bytes.
8. **Stop the engine, open *Mission replay*.** Scrub the recorded sortie at
   0.5x to 10x, with the full twin state reconstructed at every sample, plus a
   mission health report.
9. **Open *Models and interfaces*.** Held-out model metrics, the interface control
   document, and HMAC frame verification including a tampering demonstration.

---

## What makes this more than a dashboard

**The twin estimates state; it does not look values up.** Each channel is tracked
by a Kalman filter whose process model is the engine's real thermal lag. Detection
uses the innovation divided by its own standard deviation, so residuals are
statistically normalised rather than scaled by hand-tuned constants.

| Comparison method | Worst deviation, rapid throttle transitions |
| --- | --- |
| Steady-state expectation vs measurement | over 100 sigma |
| Kalman innovation | under 6 sigma |

Measured result: **zero alarms across 1250 healthy samples** on all five mission
profiles, while every injected fault is still caught in 3 to 75 seconds.

**Faults act on physics, not on readings.** A misfire sets one cylinder's
combustion quality to zero. Overheating reduces cooling airflow. Coking raises
intake restriction. The symptoms then propagate through the same model the twin
uses, so cross-channel consistency is real rather than scripted. Sensor drift is
the only sensor-side fault, because that is the case which defeats thresholds.

**Vibration is order-tracked, not a scalar.** A 2 kHz accelerometer block is
synthesised each sample and FFT-analysed against shaft speed. Half crank order
identifies a single dead cylinder on a four-stroke, first order identifies
imbalance, and a 500-950 Hz band tracks bearing and lubrication distress.

**Per-cylinder probes localise the fault.** A dead cylinder runs cold; a lean
injector runs hot. One EGT probe per cylinder is what separates them, and the
dashboard shows which cylinder.

**Detection fuses three independent detectors.** A model-based statistical test
that needs no training, an unsupervised novelty detector trained only on healthy
data so unlabelled failure modes still raise a flag, and a supervised classifier
for naming the fault. A transparent physics prior is blended in and kept visible,
so disagreement between the learned model and physical reasoning is surfaced
rather than hidden.

With no trained model present the deterministic path alone still detects and names
eight of the nine faults, with no false alarms. Only sensor drift needs the
learned model. That separation is deliberate: the safety-relevant capability does
not depend on a machine-learned component.

**RUL is extrapolated, not rescaled.** Each subsystem's health trend is regressed
against engine hours and projected to the failure threshold, with the slope's
standard error propagated into a 95 % confidence interval and the whole thing
capped by hours remaining to TBO.

**Explanations are computed.** Attribution comes from occluding each physically
meaningful feature group and measuring the drop in the trained model's class
probability, then pairing each contribution with the measured value, the
expectation, and the direction of deviation, so it can be audited against the
telemetry.

---

## Verification

```powershell
pytest                    # 215 tests, about 12 minutes
pytest -m "not slow"      # fast subset, about 1 minute
```

The suite checks the physics against closed-form relationships (ideal gas law,
Otto cycle efficiency, four-stroke induction flow, the peak-EGT mixture curve),
the estimator's transient behaviour, fault signature separation, CAN round-trip
fidelity, HMAC tampering detection, database migration safety, and the full API
including authentication and mission replay.

Three tests exist specifically to make label leakage structurally impossible:
`Diagnostics.analyse` has no parameter that could carry the ground truth, no
feature name references the label, and the telemetry frame carries no fault
information.

---

## Held-out model performance

Split by session with `GroupShuffleSplit`, 33 sessions never seen in training.
Consecutive samples within a run are strongly autocorrelated, so a random split
would leak near-identical rows across the boundary.

| Metric | Value |
| --- | --- |
| Accuracy | 0.981 |
| Macro F1 | 0.977 |
| Weakest class | `sensor_drift`, F1 0.909 |
| Novelty false alarm on healthy data | 1.0 % |

End-to-end through the full pipeline with alarm debouncing:

| Measurement | Result |
| --- | --- |
| False alarms on a healthy engine, all five profiles | 0 of 1250 samples |
| Detection latency | 3 s (misfire) to 75 s (sensor drift) |
| Correct identification once developed | 100 %, all nine faults |

Full detail, the threshold trade-off table and known limitations are in
`docs/model_card.md`. Live values are served at `GET /model/metrics`.

---

## Monitored parameters

Speed, cylinder head temperature (bulk and per cylinder), exhaust gas temperature
(bulk and per cylinder, with spread and hottest/coldest cylinder), oil pressure
and temperature, fuel flow, air/fuel ratio, manifold pressure, injection advance,
vibration level with crank-order decomposition, battery voltage, alternator
current, shaft power, ambient conditions, density ratio, pressure altitude and
accumulated engine hours.

## Fault modes

`misfire`, `injector_abnormality`, `overheating`, `lubrication_issue`,
`excessive_vibration`, `sensor_drift`, `combustion_instability`,
`coking_degradation`, `alternator_failure`.

## Mission profiles

`normal_cruise`, `high_altitude`, `hot_weather`, `long_endurance`,
`rapid_throttle_transition`.

---

## Security

| Control | Implementation |
| --- | --- |
| Authentication | `X-API-Key`, constant-time comparison, required for every state-changing route and the WebSocket |
| Telemetry integrity | HMAC-SHA256 over canonical JSON of each frame, verifiable at `POST /security/verify` |
| Audit | every control action recorded with actor, action, detail and outcome |
| Fail-safe | refuses to start with default or disabled credentials when `TRINETRA_ENV=production` |

Configure with `TRINETRA_API_KEY`, `TRINETRA_HMAC_KEY` and `TRINETRA_AUTH`. The
frontend reads `VITE_TRINETRA_API_KEY`. Transport encryption is not terminated
here; deploy behind TLS.

---

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRINETRA_API_KEY` | `trinetra-dev-key` | REST and WebSocket authentication |
| `TRINETRA_HMAC_KEY` | `trinetra-dev-hmac` | telemetry frame signing |
| `TRINETRA_AUTH` | `1` | set `0` to disable authentication |
| `TRINETRA_ENV` | `development` | `production` rejects default credentials |
| `TRINETRA_INGESTION` | `direct` | `direct` or `can` |
| `TRINETRA_CAN_INTERFACE` | `virtual` | `socketcan` with `can0` for hardware |
| `TRINETRA_CAN_CHANNEL` | `trinetra` | CAN channel name |
| `TRINETRA_DB` | `data/trinetra.db` | SQLite location |

---

## Documentation

| Document | Contents |
| --- | --- |
| `docs/architecture.md` | layered architecture, estimator design, fault-to-physics mapping, analytical redundancy |
| `docs/interface_control.md` | engine data bus ICD: PGNs, signal layouts, resolutions, transducer accuracies |
| `docs/model_card.md` | model provenance, evaluation protocol, results, limitations |
| `docs/deployment_roadmap.md` | rig correlation through fleet deployment, certification position, engineering debt |
| `brain.md` | orientation for anyone changing the code |

Regenerate the ICD after changing the frame table:

```powershell
python -m tools.generate_icd_doc
```

---

## Repository layout

```
backend/
  config/            engine specification, thresholds, credentials
  simulator/         physics core, vibration synthesis, stateful engine, dataset generator
  signal_processing/ order tracking and condition indicators
  ingestion/         source abstraction, J1939-style frames, CAN transport
  digital_twin/      Kalman estimator, twin state, synchroniser
  preprocessing/     the single feature definition used by training and serving
  ml/                training, diagnostics fusion, RUL
  xai/               grouped occlusion attribution
  security/          API key guard, HMAC signing, audit
  database/          missions, snapshots, audit trail
  services/          acquisition cycle to snapshot
  api/               control, telemetry, missions, system routers
frontend/src/
  services/          API client and WebSocket
  components/        primitives, charts, panels
  pages/             dashboard, replay, models and interfaces
docs/                architecture, ICD, model card, roadmap
tools/               ICD document generator
tests/               215 tests
```
