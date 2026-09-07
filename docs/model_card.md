# Model Card - Diagnostic Models

Regenerate the numbers with:

```
python -m backend.simulator.generate_dataset
python -m backend.ml.train
```

Live values are served at `GET /model/metrics` and written to `models/metrics.json`.

---

## Intended use

Detect and identify degradation and fault conditions in a **simulated** aero
piston engine, for research, demonstration and architecture evaluation.

**Not** for airworthiness, dispatch, maintenance release or any decision
affecting a real aircraft. The training data is synthetic and the model has never
seen a real engine.

---

## Models

| Model | Purpose | Training data | Key hyperparameters |
| --- | --- | --- | --- |
| `IsolationForest` | unsupervised novelty; covers failure modes that were never labelled | healthy samples only | 300 trees, contamination 0.02 |
| `RandomForestClassifier` | fault identification with class probabilities | all 10 classes | 400 trees, min_samples_leaf 2, balanced_subsample |
| `StandardScaler` | feature scaling | training split only | fitted before either model |

A third detector, the mean normalised innovation squared from the Kalman twin, is
model-based and requires no training. All three are fused in
`backend/ml/diagnostics.py`.

---

## Data

| Property | Value |
| --- | --- |
| Samples | 31,000 |
| Features | 93 |
| Classes | 10 (`normal` plus 9 fault modes) |
| Sessions | 130 independent runs |
| Mission profiles | normal cruise, high altitude, hot weather, long endurance, rapid throttle transition |
| Generator | `backend/simulator/generate_dataset.py` |

Each session warms the engine, lets the filters converge, then injects one fault
and records its whole development from incipient to fully expressed, so the
classifier sees early-stage faults and not only obvious ones. Six extra
envelope-randomised healthy sessions per profile densify the healthy manifold,
because the novelty detector's decision boundary is drawn entirely from it.

The rapid throttle transition profile is included deliberately so the classifier
learns that a healthy transient is not a fault.

---

## Evaluation protocol

Split by **session identifier** using `GroupShuffleSplit`, 25 % held out
(97 training sessions, 33 test sessions).

Consecutive samples within one run are strongly autocorrelated. A plain random
split would place nearly identical rows on both sides of the boundary and report
an accuracy that does not survive contact with new data. No session contributes
to both sides.

---

## Results on held-out sessions

| Metric | Value |
| --- | --- |
| Accuracy | 0.981 |
| Macro F1 | 0.977 |

Per-class F1:

| Class | F1 |
| --- | --- |
| misfire | 1.000 |
| excessive_vibration | 0.999 |
| alternator_failure | 0.998 |
| combustion_instability | 0.997 |
| injector_abnormality | 0.996 |
| lubrication_issue | 0.981 |
| normal | 0.979 |
| coking_degradation | 0.969 |
| overheating | 0.943 |
| sensor_drift | 0.909 |

Novelty detector threshold trade-off, measured on held-out healthy data, with
detection scored only where fault severity exceeds 0.6:

| Healthy quantile | False alarm rate | Developed-fault detection |
| --- | --- | --- |
| p95 | 0.050 | 0.676 |
| p98 | 0.020 | 0.580 |
| **p99 (operating point)** | **0.010** | **0.491** |
| p99.5 | 0.005 | 0.414 |
| p99.9 | 0.001 | 0.270 |

The conservative operating point is a deliberate choice: novelty exists to catch
unlabelled behaviour, while sensitivity is carried by the statistical and
supervised detectors. Read on its own the detection column looks weak; in the
fused system it is a safety net, not the primary detector.

### End-to-end behaviour

Measured through the full pipeline with alarm debouncing, which is what actually
matters. These numbers are not from the static test matrix.

| Measurement | Result |
| --- | --- |
| False alarms, healthy engine, all five profiles | 0 of 1250 samples |
| Detection latency | 3 s (misfire, vibration, alternator) to 75 s (sensor drift) |
| Correct identification once developed | 100 % of samples, all nine faults |
| Limiting subsystem correctly attributed | 9 of 9 faults |

Per-fault detection latency at cruise:

| Fault | Detected | Fault | Detected |
| --- | --- | --- | --- |
| misfire | 3 s | coking_degradation | 25 s |
| excessive_vibration | 4 s | overheating | 36 s |
| alternator_failure | 4 s | sensor_drift | 75 s |
| combustion_instability | 7 s | | |
| injector_abnormality | 11 s | | |
| lubrication_issue | 19 s | | |

---

## Most influential features

| Feature | Importance | Reading |
| --- | --- | --- |
| `alternator_output_isolation` | 0.037 | electrical bus analytical redundancy |
| `vib_order_0p5_ratio` | 0.036 | half crank order, single-cylinder misfire |
| `vib_order_1p5_ratio` | 0.035 | 1.5x sideband, combustion irregularity |
| `cht_drift` | 0.034 | learned slow bias on head temperature, the sensor-drift signature |
| `vib_imbalance_index` | 0.034 | first-order to firing-order ratio, rotating imbalance |
| `vib_order_1p0_ratio` | 0.034 | first order amplitude |
| `vibration_nres` | 0.034 | overall vibration innovation |
| `alternator_output_nres` | 0.032 | alternator current innovation |
| `vib_misfire_index` | 0.032 | half-order to firing-order ratio |
| `oil_pressure_cond_nres` | 0.029 | pressure against the temperature-conditioned expectation |
| `power_kw_rel` | 0.028 | shaft power relative to expectation |

The ranking is dominated by physically interpretable quantities rather than raw
readings, which is a consequence of feeding the models normalised innovations and
crank-order ratios instead of gauge values.

---

## Feature construction and leakage controls

Features are defined once in `backend/preprocessing/pipeline.py` and that same
code is driven by both the dataset generator and the live service, so train/serve
skew has no way in.

Structural guarantees, each covered by a test in `tests/test_diagnostics.py`:

- `Diagnostics.analyse(features, twin)` has no parameter that could carry the
  injected fault, severity or simulator wear state.
- `FeatureBuilder.transform(telemetry, twin, vibration, cylinders)` likewise.
- No feature name contains `fault`, `label`, `truth`, `severity`, `wear`,
  `injected` or `mission`.
- The telemetry frame itself carries no fault information. Ground truth travels in
  a separate `simulated_truth` field on the snapshot, for demonstration and
  offline scoring only.

Features are dimensionless wherever possible - normalised innovations, crank
orders referenced to the firing order, ratios against expectation - so a model
trained at one operating point stays valid across the envelope.

---

## Known limitations

- **Synthetic data.** Performance reflects the simulator's fault physics. A real
  engine will show correlations, sensor faults and failure modes not represented
  here, and these numbers should not be expected to transfer.
- **Sensor drift is the weakest class** (F1 0.909, 75 s latency). It is the
  hardest case by construction: the deviation is slow, small per sample, and
  partly absorbed by baseline adaptation before it becomes conspicuous. It is
  also the one fault the physics-rule fallback cannot name on its own.
- **Overheating F1 is 0.943**, the second weakest. Its early stage overlaps with
  a hot-weather cruise, which is exactly the ambiguity the mission profile
  context features exist to resolve, and they do not resolve it perfectly.
- **Single fault at a time.** The classifier is trained on one active fault per
  session and is not validated on simultaneous faults.
- **One engine configuration.** Trained for a single 1.35 L four-cylinder
  specification; another engine needs retraining.
- **No calibration guarantee on probabilities.** Confidence is a fused score, not
  a calibrated posterior. It should be read as a ranking, not as a probability of
  being correct.
- **Version pinning.** scikit-learn does not guarantee unpickling across minor
  releases, hence the pinned `requirements.txt`. The loader validates the feature
  schema and falls back to physics rules if it has changed.

---

## Graceful degradation

If the model bundle is missing, unreadable or built against a different feature
schema, `Diagnostics` reports `model_backend: physics_prior_fallback` and
continues on the transparent physics rules and the statistical residual tests
alone. It does not silently claim to be a trained model.

Measured fallback capability, same conditions as above:

| Outcome | Faults |
| --- | --- |
| Detected and correctly named | 8 of 9 (misfire, injector, overheating, lubrication, coking, combustion instability, excessive vibration, alternator) |
| Not detected | sensor drift |
| False alarm on a healthy engine | none |

So the deterministic path alone covers everything except the fault that depends
most on learned behaviour. That matters for the certification argument in
`docs/deployment_roadmap.md`: the safety-relevant detection capability does not
require the trained model.
