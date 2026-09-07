# Digital Twin Architecture

Trinetra AI mirrors a MALE-UAV aero piston engine as a continuously synchronised
virtual engine, and turns the difference between the two into health indices,
diagnoses, remaining-life estimates and dispatch advice.

Everything in this repository runs on **synthetic telemetry**. Nothing is
calibrated against a real engine and nothing here is certified for flight,
maintenance or airworthiness decisions.

---

## 1. Layered view

```
                      ENGINE / TEST RIG                     EDGE NODE
        +-------------------------------+     +-------------------------------+
        |  Aero piston engine           |     | Accelerometer sampling 2 kHz  |
        |  ECU / FADEC                   |     | Hann-window FFT               |
        |  Thermocouples, transducers    |---->| RPM-synchronous order track   |
        |  Accelerometer                 |     | Condition indicators only     |
        +-------------------------------+     +---------------+---------------+
                                                              |
                                       CAN 2.0B, J1939-style, 250 kbit/s
                                       11 frames / 88 bytes per sample
                                                              v
    +-------------------------------------------------------------------------+
    |                      GROUND CONTROL STATION / SERVER                    |
    |                                                                         |
    |  ingestion/        base.TelemetrySource  -> DirectSource | CanBusSource  |
    |        |                                                                |
    |        v                                                                |
    |  simulator/physics EnginePhysics: thermodynamic model (shared reference) |
    |        |                                                                |
    |        v                                                                |
    |  digital_twin/     expected state -> Kalman bank -> innovations         |
    |                    subsystem health, analytical redundancy              |
    |        |                                                                |
    |        v                                                                |
    |  preprocessing/    windowed, dimensionless feature vector (93)          |
    |        |                                                                |
    |        +--------------------+--------------------+                      |
    |        v                    v                    v                      |
    |  statistical NIS      IsolationForest      RandomForest                 |
    |  (model based)        (novelty, healthy    (fault identity)             |
    |                        data only)                                       |
    |        \                    |                    /                      |
    |         +-------------------+-------------------+                       |
    |                             v                                          |
    |  ml/diagnostics    fusion + physics prior + alarm debouncing            |
    |        |                                                                |
    |        +---------------> ml/rul       trend extrapolation + 95 % CI     |
    |        +---------------> xai/explainer grouped occlusion attribution    |
    |        |                                                                |
    |        v                                                                |
    |  services/         snapshot assembly, HMAC signing, mission recording   |
    |        |                                                                |
    |        +--> REST + WebSocket  --> React dashboard (live / replay)       |
    |        +--> SQLite            --> missions, snapshots, audit trail      |
    +-------------------------------------------------------------------------+
```

---

## 2. Why the twin is a state estimator, not a lookup

The naive approach compares each measurement against a steady-state expectation.
That fails during every throttle transient, because a cylinder head has a time
constant of roughly 20 s and oil temperature over 60 s: the engine is legitimately
far from its steady-state target while it catches up.

Each channel is therefore tracked by a scalar Kalman filter whose process model is
the engine's own first-order lag:

```
x[k+1] = x[k] + (u[k] - x[k]) * dt/tau + w      w ~ N(0, Q)
z[k]   = x[k] + v                               v ~ N(0, R)
```

`u` is the steady-state target from the thermodynamic model, computed from the
*measured* throttle and atmospheric conditions. `tau` encodes how fast the
physical channel is allowed to move. `R` is the transducer accuracy declared in
the interface control document.

Detection uses the **innovation** (measurement minus one-step prediction) divided
by the innovation standard deviation, which yields a statistically normalised
residual instead of a hand-tuned scale factor. Measured on the rapid-throttle
mission profile:

| Comparison method | Worst deviation during transients |
| --- | --- |
| Steady-state expectation vs measurement | over 100 sigma |
| Kalman innovation | under 6 sigma |

That difference is the reason the system produces **zero alarms across 1250
healthy samples** spanning all five mission profiles, while still detecting every
injected fault within 3 to 75 seconds.

### Gradual drift versus abrupt fault

Each filter also maintains a slowly adapting bias, capped at 2.5 sigma and frozen
whenever the normalised innovation exceeds 4 sigma. Build tolerance, calibration
offset and gradual wear are absorbed into that bias and reported as
**degradation**; whatever remains in the innovation is reported as an **anomaly**.
This is what lets a slow coking trend be told apart from an abrupt misfire.

---

## 3. Faults act on physics, not on readings

A fault never edits an output value. It changes a physical parameter, and the
symptom propagates through the same model the twin uses. Cross-channel
consistency is therefore real rather than scripted.

| Fault | Parameter actually perturbed | Emergent symptoms |
| --- | --- | --- |
| Misfire | one cylinder's combustion quality to zero | half-crank-order energy, that cylinder's EGT collapses, shaft power falls |
| Injector abnormality | one cylinder's metering error, lean | that cylinder runs hot (late burn), EGT spread rises, power falls |
| Overheating | cooling airflow reduced (blocked duct) | CHT and oil temperature rise together, oil pressure follows viscosity |
| Lubrication issue | oil leakage and cooler effectiveness | pressure below the temperature-conditioned expectation, broadband vibration |
| Excessive vibration | rotating imbalance | first-order energy only, no thermal change |
| Combustion instability | cycle-to-cycle combustion quality and mixture | oscillating EGT and mixture, sidebands at 1.5x |
| Coking degradation | intake restriction, so volumetric efficiency falls | speed, airflow and fuel flow all down at unchanged throttle |
| Alternator failure | alternator field collapse | output current falls, battery discharges |
| Sensor drift | **sensor-side bias only, after the physics** | CHT reads high while oil temperature and EGT do not move |

Sensor drift is deliberately the only sensor-side fault. It is the case that
defeats threshold monitoring, because a drifting CHT probe and genuine
overheating look identical on that one gauge.

---

## 4. Analytical redundancy

Channels are grouped by physical coupling. Within each group the median deviation
is the consensus, and a member far above that consensus is inconsistent with the
group's physics, which points at the instrument rather than the engine.

| Group | Members | Physical reason they move together |
| --- | --- | --- |
| Thermal chain | CHT, oil temperature | same heat rejection path |
| Combustion chain | EGT, fuel flow, air/fuel ratio | exhaust temperature is set by mixture and fuel rate |
| Load chain | RPM, manifold pressure, fuel flow | all follow delivered power |
| Lubrication chain | oil pressure, oil temperature | pressure follows viscosity |
| Electrical bus | alternator output, battery voltage | one charges the other |

Plausibility is reported and used as a model feature, but it is **not** an alarm
source on its own. During a real fault one channel legitimately leads its group,
so isolation alone would misattribute engine faults to instrumentation.

### Cascaded residual

Oil pressure is evaluated twice: against the nominal expectation, and against
what the pump should deliver at the **measured** oil temperature. The second
residual is what drives the lubrication index. Without it, overheating is blamed
on the oil system for a pressure drop that hot, thin oil fully explains.

---

## 5. Detection and identification

Three independent detectors, fused:

| Detector | Basis | Covers | Operating point |
| --- | --- | --- | --- |
| Statistical, aggregate | mean normalised innovation squared | broad departures from physics; needs no training | limit 9.0 |
| Statistical, per channel | worst single-channel NIS | faults confined to one channel, which the aggregate dilutes | limit 16.0 (4 sigma) |
| Unsupervised | IsolationForest on healthy data only | unfamiliar and unlabelled failure modes | 1.0 % false alarm on held-out healthy data |
| Supervised | RandomForest class probabilities | naming the fault | 0.981 accuracy, 0.977 macro F1 |

Two statistical tests are needed rather than one. The aggregate averages over
roughly two dozen channels, so a fault confined to a single channel barely moves
it: an imbalance producing a 1.5 sigma vibration innovation contributes about 0.09
to the mean. The per-channel test catches exactly that case. On healthy data the
worst channel peaks around 9.9 against a limit of 16.

### Which channels count toward the statistic

NIS is a filter-consistency measure, so it is only valid where the process model
is unbiased. The nominal oil-pressure channel predicts from the *steady-state*
oil temperature, which an engine on a 40 s throttle cycle never reaches, so it
carries a known transient bias. Measured on the transition profile it contributed
a mean NIS of 175 while every other channel stayed below 1.1. It is therefore
excluded from the aggregate (`NIS_EXCLUDED` in `twin.py`) while remaining a
reported value and a model feature, and the temperature-conditioned channel
provides the unbiased version.

Identification blends the classifier probabilities (75 %) with a transparent
physics-rule prior (25 %). The prior stays visible in the output, so an engineer
can see when the learned model and physical reasoning **disagree** — which is
exactly the case that deserves human attention.

Alarms are debounced: a trigger must hold for 4 of the last 6 samples. A single
sample crossing a threshold is not an alarm, which is the usual reason operators
end up switching a monitoring system off.

---

## 6. Remaining useful life

RUL is not a rescaled health number. Each subsystem's smoothed health index is
regressed against accumulated engine hours and extrapolated to the failure
threshold:

```
RUL = (health_now - health_failure) / -slope
```

The slope's standard error propagates into a 95 % confidence interval, so the
result is a band rather than a single misleading figure. The overall figure is set
by the limiting subsystem and capped by hours remaining to published TBO. Before
45 samples the estimator reports that it is still establishing a trend instead of
inventing a number.

Wear itself accumulates from operating stress, not a fixed counter: Arrhenius-style
temperature acceleration (every ~28 K above reference roughly doubles thermal
wear), cubic speed dependence for mechanical wear, and low-pressure penalties for
the oil system.

---

## 7. Edge/ground split

The raw accelerometer stream stays on the edge node. A 2 kHz waveform needs
roughly 33 kB/s, which is not a reasonable load for a shared 250 kbit/s engine
bus. FFT and RPM-synchronous order tracking run next to the engine, and only the
derived condition indicators are transmitted — three frames per sample.

This is why the CAN interface carries `vib_order_0p5` through `vib_order_4p0`,
crest factor and spectral kurtosis rather than samples.

---

## 8. Module map

| Path | Responsibility |
| --- | --- |
| `backend/config/settings.py` | engine specification, thresholds, credentials, paths |
| `backend/simulator/physics.py` | thermodynamic model; the shared reference for engine and twin |
| `backend/simulator/vibration.py` | crank-order waveform synthesis; healthy driver baseline |
| `backend/simulator/engine.py` | stateful engine: thermal lag, fault parameters, wear |
| `backend/signal_processing/spectral.py` | order tracking and condition indicators |
| `backend/ingestion/` | source abstraction, J1939-style frames, CAN transport |
| `backend/digital_twin/estimator.py` | Kalman bank, innovations, baseline adaptation |
| `backend/digital_twin/twin.py` | expected state, subsystem health, analytical redundancy |
| `backend/preprocessing/pipeline.py` | the single feature definition used by training and serving |
| `backend/ml/train.py` | grouped-split training, metrics, model bundle |
| `backend/ml/diagnostics.py` | detector fusion, physics prior, debouncing |
| `backend/ml/rul.py` | degradation trends and maintenance advisories |
| `backend/xai/explainer.py` | grouped occlusion attribution and mechanism narrative |
| `backend/security/` | API key guard, HMAC frame signing, audit |
| `backend/database/repository.py` | missions, snapshots, audit; migration-safe |
| `backend/services/engine_service.py` | one acquisition cycle to one snapshot |
| `backend/api/` | control, telemetry, missions, system routers |
| `frontend/src/` | live dashboard, mission replay, model and interface views |

---

## 9. Preventing train/serve skew and label leakage

Two structural rules, both covered by tests:

1. **One feature definition.** `backend/preprocessing/pipeline.py` is driven by
   both the offline dataset generator and the live service. There is no second
   copy to fall out of step.
2. **No path for the label.** `Diagnostics.analyse(features, twin)` has no
   parameter that could carry the injected fault, severity or simulator wear
   state. `FeatureBuilder.transform` likewise takes only telemetry, twin state,
   vibration and cylinder data. Ground truth travels in a separate
   `simulated_truth` field on the snapshot, used for demonstration and scoring
   only.

Evaluation splits by **session** using `GroupShuffleSplit`. Consecutive samples in
one run are strongly autocorrelated, so a plain random split would leak nearly
identical rows across the boundary and report an accuracy that does not survive
contact with new data.
