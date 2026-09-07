# Trinetra AI - Project Brain

Orientation for anyone changing this code. Read `docs/architecture.md` for the
design reasoning; this file is about where things live and what will break.

## Purpose and safety boundary

A digital twin for a MALE-UAV-class aero piston engine: thermodynamic model
synchronised to telemetry by Kalman filtering, subsystem health, fault detection
and identification, RUL, mission replay and an operator dashboard.

**Every value is synthetic.** The physics is dimensionally realistic and
first-principles, but it is not calibrated against any real engine. Nothing here
is valid for flight, maintenance or airworthiness decisions. Keep the disclaimers
in the API, the UI and generated documents intact.

## Stack

| Area | Technology | Entry point |
| --- | --- | --- |
| API | FastAPI, Uvicorn | `backend/main.py` |
| Live stream | WebSocket, authenticated | `ws://localhost:8000/ws/telemetry?api_key=...` |
| Numerics | NumPy | `backend/simulator/physics.py` |
| Models | scikit-learn, joblib | `models/diagnostics.joblib` |
| Bus | python-can (virtual or socketcan) | `backend/ingestion/sources.py` |
| Persistence | SQLite (WAL) | `data/trinetra.db` |
| Dashboard | React 18, Vite, Recharts | `frontend/src/main.jsx` |
| Tests | pytest, 215 tests | `tests/` |

## Data flow

```
TelemetrySource.read()                      DirectSource | CanBusSource
  -> IngestedFrame(telemetry, cylinders, vibration, waveform, truth)
  -> DigitalTwin.state()                    physics expectation + Kalman innovations
  -> FeatureBuilder.transform()             93 dimensionless features
  -> Diagnostics.analyse()                  3 detectors fused + physics prior + debounce
  -> RulEstimator.update()                  trend extrapolation + 95 % CI
  -> Explainer.explain()                    grouped occlusion attribution
  -> EngineService.tick()                   snapshot + HMAC signature
       -> in-memory history, SQLite, WebSocket subscribers, REST
```

`EngineService.tick()` is the only place one acquisition cycle becomes one
snapshot. `TwinSynchroniser` binds twin and features and is shared by the live
service and the dataset generator, which is what prevents train/serve skew.

## Two invariants that must not be broken

**1. No label reaches inference.** `Diagnostics.analyse(features, twin)` and
`FeatureBuilder.transform(telemetry, twin, vibration, cylinders)` have no
parameter that could carry the injected fault, severity or simulator wear. Ground
truth rides in `snapshot['simulated_truth']` for demonstration only. Three tests
in `tests/test_diagnostics.py` enforce this structurally. If you add a parameter
to either signature, those tests will fail, and that is the point.

**2. One feature definition.** `backend/preprocessing/pipeline.py` is driven by
both `generate_dataset.py` and the live service. Never add a second feature path.

## Changing the feature set

Features changed means the model must be rebuilt. `FEATURE_NAMES` is stored in the
bundle and validated on load; a mismatch drops to `physics_prior_fallback` rather
than producing nonsense.

```powershell
python -m backend.simulator.generate_dataset   # ~2 min, 31k rows
python -m backend.ml.train                     # ~2 min
```

Also update, in the same change: `backend/xai/explainer.py` `GROUPS` (a test
asserts every feature belongs to a group), and the physics prior in
`diagnostics.py` if the channel it references was renamed.

## Adding a telemetry channel

Coherent change means touching all of these:

1. `physics.steady_state()` - the expected value. Without it the estimator
   silently skips the channel; that bug shipped once already for `vibration`.
2. `simulator/engine.py` - `Telemetry` field, thermal lag in `TAU` if it has
   inertia, sensor noise in the `noise` dict.
3. `digital_twin/estimator.py` - `CHANNEL_MODELS` entry with `tau`, `q`, `r`.
   `r` must be the squared 1-sigma transducer accuracy from the ICD.
4. `digital_twin/twin.py` - `SUBSYSTEM_CHANNELS` weight and direction if it feeds
   a health index, `COUPLED_GROUPS` if it is physically coupled to others.
5. `preprocessing/pipeline.py` - `RESIDUAL_CHANNELS` and `ROC_SCALE`.
6. `ingestion/can_frames.py` - a signal in a message, then regenerate the ICD.
7. `xai/explainer.py` - `GROUPS` and `UNITS`.
8. Frontend `TELEMETRY_FIELDS` in `pages/Dashboard.jsx`.
9. Regenerate the dataset and retrain.

## Subsystem attribution uses direction

`SUBSYSTEM_CHANNELS` maps channel to `(weight, direction)` where `+1` counts only
positive deviations, `-1` only negative, `0` absolute. This matters: a dead
cylinder runs **cold**, so cold cylinder temperatures must feed combustion, not
thermal. Getting this wrong makes a misfire look like an overheat.

Oil pressure is evaluated twice. `oil_pressure` is against the nominal
expectation; `oil_pressure_cond` is against what the pump should deliver at the
**measured** oil temperature, and that second one drives the lubrication index.
Without the split, overheating gets blamed on the oil system for a pressure drop
that hot oil fully explains. Its `q` is deliberately large because the viscosity
correlation is fitted near normal temperature and extrapolates poorly.

## Baseline adaptation

Each filter learns a slow bias, capped at 2.5 sigma, frozen above 4 sigma of
normalised innovation. Build tolerance and gradual wear are absorbed and reported
as degradation; the remainder is an anomaly. Widening `BASELINE_LIMIT` hides real
faults. Narrowing it makes normal wear read as a fault. `cht_drift` is the
top-ranked model feature precisely because it captures this, so changing the cap
changes model behaviour.

## Alarm debouncing

`Diagnostics` is stateful: a trigger must hold for 4 of the last 6 samples. One
instance per session. Sharing an instance across sessions mixes their histories.
`EngineService.reset()` rebuilds it, along with the explainer that wraps it.

Sensor plausibility is reported but is **not** an alarm trigger. During a real
fault one channel legitimately leads its coupled group, so isolation alone
misattributes engine faults to instrumentation.

## API surface

Routers in `backend/api/`. Control routes require `X-API-Key`; reads are open.

- `control.py` - `/simulation/{start,stop,reset,fault}`, `/mission/configure`,
  `/ingestion/mode`
- `telemetry.py` - `/telemetry/*`, `/digital-twin/*`, `/prediction`, `/fault`,
  `/rul`, `/explanation`, `/advisory`, `/vibration/spectrum`, `/alarms`,
  `/ingestion/*`
- `missions.py` - list, get, `/timeline`, `/snapshots`, `/report`, delete
- `system.py` - `/health`, `/system/capabilities`, `/model/*`,
  `/interface-control`, `/security/*`

Snapshot shape: `sequence, mission_id, mission_profile, received_at, telemetry,
cylinders, vibration, digital_twin, prediction, rul, explanation, advisory,
ingestion, integrity, simulated_truth`. The frontend reads these keys directly, so
renaming one is a breaking change on both sides.

## Persistence

Missions own snapshots. `/simulation/reset` clears live state and **preserves**
stored missions; `DELETE /missions` is the destructive one and is audited.

`Repository._migrate()` renames a pre-mission `snapshots` table to
`snapshots_legacy_vN` rather than dropping it. Do not turn that into a drop.

One connection with `check_same_thread=False` and a mutex. Adequate for a single
process, wrong for multi-worker serving.

## Frontend

- `services/api.js` - the only place that knows the base URL and API key.
  Configure with `VITE_TRINETRA_API` and `VITE_TRINETRA_API_KEY`.
- `components/` - `primitives.jsx`, `charts.jsx`, `panels.jsx`
- `pages/` - `Dashboard.jsx`, `Replay.jsx`, `SystemInfo.jsx`
- Tabs are local state, no router dependency.
- `npm run build` is `vite build`. It used to be plain `vite`, which started a dev
  server and hung any build script.

## Testing

```powershell
pytest                  # everything, ~12 min
pytest -m "not slow"    # ~1 min
```

`pytest.ini` disables `pytest_ethereum`, an unrelated broken plugin present in
this machine's user site-packages that otherwise breaks collection.

`tests/conftest.py` redirects `TRINETRA_DB` to a temporary directory **before**
importing any backend module, because `Settings` reads the environment at import.
It also exposes `run_pipeline()`, which most tests use instead of rebuilding the
chain.

Prefer paired control runs over pre-injection baselines when asserting fault
effects. Slow channels such as oil temperature (tau 65 s) are still warming after
a 120-sample warmup, and comparing against a same-seed healthy control isolates
the fault from that drift. `tests/test_simulator.py::paired` does this.

## Calibration constants

In `physics.py`: `CYCLE_DEVIATION`, `VE_MAX`, `VE_PEAK_RPM`, `VE_SPREAD_RPM`,
`COOLING_COEFF`, `OIL_PUMP_PSI_REF`, `EXHAUST_HEAT_FRACTION`,
`HEAD_HEAT_FRACTION`, `LEAN_LIMIT_AFR`. They are chosen so the model reproduces
the reference engine's published rating and typical cruise instrument readings.
`tests/test_physics.py::test_instrument_readings_land_in_realistic_ranges` guards
against a refactor quietly producing unphysical gauge values, and the mixture
sweep test guards the peak-EGT shape.

Changing any of them changes the training distribution. Regenerate and retrain.

## Known limitations

- `WEAR_ACCELERATION = 1800` makes one simulated second worth thirty engine
  minutes so degradation is visible in a demonstration. It is not a real hours
  counter.
- The estimator assumes a regular 1 Hz cadence. Missing or out-of-order frames
  are not handled.
- No sensor range or validity gating on ingest. A transducer failed to full scale
  reaches the filters.
- Single active fault only; simultaneous faults are untrained and unvalidated.
- Fault injection endpoints must be removed from any production build, not merely
  access-controlled.
- `docs/interface_control.md` is generated. Edit `can_frames.py` then run
  `python -m tools.generate_icd_doc`.

## Do not commit

`.venv/`, `frontend/node_modules/`, `frontend/dist/`, `__pycache__/`,
`.pytest_cache/`, runtime `data/trinetra.db` contents. `models/*.joblib` is a
build artifact reproducible from the training commands; commit it only if a
reviewer needs to run without training first.
