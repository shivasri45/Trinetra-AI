"""Detection, identification and the guarantee that no label reaches inference."""
from __future__ import annotations

import inspect

import pytest

from backend.config.settings import FAULTS, MISSION_PROFILES
from backend.digital_twin.synchroniser import TwinSynchroniser
from backend.ml.diagnostics import Diagnostics
from backend.preprocessing.pipeline import FEATURE_NAMES, FeatureBuilder
from backend.simulator.engine import EngineSimulator
from conftest import run_pipeline

FAULT_LIST = [f for f in FAULTS if f != 'normal']


# ------------------------------------------------------------- leakage guards
def test_analyse_cannot_receive_the_ground_truth():
    """Structural guarantee: there is nowhere to pass the injected fault."""
    parameters = set(inspect.signature(Diagnostics.analyse).parameters) - {'self'}
    assert parameters == {'features', 'twin'}
    for banned in ('injected_fault', 'fault', 'label', 'truth', 'severity'):
        assert banned not in parameters


def test_feature_names_contain_no_label_or_simulator_state():
    banned = ('fault', 'label', 'truth', 'severity', 'wear', 'injected', 'mission')
    offenders = [name for name in FEATURE_NAMES if any(word in name for word in banned)]
    assert not offenders, offenders


def test_feature_builder_signature_excludes_truth():
    parameters = set(inspect.signature(FeatureBuilder.transform).parameters) - {'self'}
    assert parameters == {'telemetry', 'twin', 'vibration', 'cylinders'}


def test_telemetry_frame_carries_no_fault_information():
    sim = EngineSimulator(3)
    sim.inject_fault('overheating')
    frame = sim.step()
    keys = set(frame.telemetry.dict())
    assert not keys & {'fault', 'severity', 'wear', 'truth'}


def test_identical_features_give_identical_predictions_regardless_of_injection():
    """Two engines in the same physical state must be diagnosed the same way."""
    diagnostics = Diagnostics()
    frame, result, _ = run_pipeline(fault='misfire', warmup=120, after=120)
    first = diagnostics.analyse(result.features, result.twin)
    second = Diagnostics().analyse(result.features, result.twin)
    assert first['predicted_fault'] == second['predicted_fault']
    assert first['class_probabilities'] == second['class_probabilities']


# ------------------------------------------------------------------ behaviour
def test_model_bundle_is_loaded(diagnostics_available):
    if not diagnostics_available:
        pytest.skip('trained model not present; run python -m backend.ml.train')
    assert Diagnostics().backend == 'trained'


def test_healthy_engine_is_reported_normal():
    _, result, prediction = run_pipeline(warmup=180, after=120, with_diagnostics=True)
    assert prediction['predicted_fault'] == 'normal'
    assert not prediction['anomaly']
    assert prediction['anomaly_score'] < 0.5


@pytest.mark.slow
@pytest.mark.parametrize('mission', MISSION_PROFILES)
def test_no_false_alarms_on_a_healthy_engine(mission):
    """End-to-end false-alarm contract, including the throttle transition profile."""
    sim = EngineSimulator(31)
    sim.configure_mission(mission)
    sync = TwinSynchroniser()
    diagnostics = Diagnostics()
    alarms = evaluated = 0
    for step in range(340):
        frame = sim.step()
        result = sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)
        prediction = diagnostics.analyse(result.features, result.twin)
        if step > 170:
            evaluated += 1
            alarms += int(prediction['anomaly'])
    assert evaluated > 100
    assert alarms / evaluated < 0.02, f'{alarms}/{evaluated} false alarms on {mission}'


@pytest.mark.slow
@pytest.mark.parametrize('fault', FAULT_LIST)
def test_every_fault_is_detected_and_named(fault, diagnostics_available):
    if not diagnostics_available:
        pytest.skip('trained model not present')
    sim = EngineSimulator(999)
    sim.configure_mission('normal_cruise')
    sync = TwinSynchroniser()
    diagnostics = Diagnostics()
    for _ in range(160):
        frame = sim.step()
        result = sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)
        diagnostics.analyse(result.features, result.twin)

    sim.inject_fault(fault)
    detected_at = None
    correct = total = 0
    for step in range(220):
        frame = sim.step()
        result = sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)
        prediction = diagnostics.analyse(result.features, result.twin)
        if prediction['anomaly'] and detected_at is None:
            detected_at = step
        if step >= 120:
            total += 1
            correct += int(prediction['predicted_fault'] == fault)

    assert detected_at is not None, f'{fault} never detected'
    assert detected_at < 120, f'{fault} detected only after {detected_at} s'
    assert correct / total > 0.7, f'{fault} identified in only {correct}/{total} samples'


def test_single_channel_fault_is_caught_by_the_per_channel_test():
    """The aggregate dilutes a one-channel fault, so a per-channel test is needed."""
    from backend.config.settings import settings
    _, result, prediction = run_pipeline(fault='excessive_vibration', warmup=150,
                                         after=180, with_diagnostics=True)
    twin = result.twin
    assert twin['mean_nis'] < settings.nis_warning, 'aggregate should indeed be diluted'
    assert prediction['anomaly']
    assert prediction['predicted_fault'] == 'excessive_vibration'


def test_healthy_engine_stays_below_the_per_channel_limit():
    from backend.config.settings import settings
    sim = EngineSimulator(31)
    sim.configure_mission('rapid_throttle_transition')
    sync = TwinSynchroniser()
    peaks = []
    for step in range(340):
        frame = sim.step()
        result = sync.process(frame.telemetry.dict(), frame.cylinders, frame.vibration_features)
        if step > 150:
            peaks.append(result.twin['peak_nis'])
    assert max(peaks) < settings.nis_peak_warning, f'worst channel reached {max(peaks)}'


def test_detection_fuses_three_independent_detectors():
    _, result, prediction = run_pipeline(fault='lubrication_issue', warmup=150,
                                         after=180, with_diagnostics=True)
    detectors = prediction['detectors']
    assert set(detectors) == {'statistical', 'unsupervised', 'supervised'}
    assert detectors['statistical']['mean_nis'] > 0
    assert detectors['supervised']['probability'] > 0
    assert len(prediction['detection_triggers']) >= 2


def test_alarms_require_persistence():
    """A trigger present on a single sample must not raise an alarm."""
    diagnostics = Diagnostics()
    _, result, _ = run_pipeline(fault='misfire', warmup=120, after=120)
    first = diagnostics.analyse(result.features, result.twin)
    assert first['instantaneous_triggers'], 'expected instantaneous triggers on a misfire'
    assert not first['detection_triggers'], 'one sample should not confirm an alarm'
    for _ in range(diagnostics.trigger_history.maxlen):
        confirmed = diagnostics.analyse(result.features, result.twin)
    assert confirmed['detection_triggers'], 'a persistent condition should confirm'


def test_physics_prior_is_exposed_alongside_the_model():
    _, result, prediction = run_pipeline(fault='overheating', warmup=150,
                                         after=180, with_diagnostics=True)
    assert prediction['physics_prior']
    assert prediction['ml_probabilities']
    assert prediction['agreement']['state'] in ('agree', 'disagree')
    assert set(prediction['class_probabilities']) == set(FAULTS)
    assert sum(prediction['class_probabilities'].values()) == pytest.approx(1.0, abs=0.05)


def test_physics_prior_alone_still_diagnoses_without_a_model():
    """Graceful degradation when no trained bundle is available."""
    diagnostics = Diagnostics(model_dir='does-not-exist')
    assert diagnostics.backend == 'physics_prior_fallback'
    _, result, _ = run_pipeline(fault='lubrication_issue', warmup=150, after=180)
    for _ in range(8):
        prediction = diagnostics.analyse(result.features, result.twin)
    assert prediction['model_backend'] == 'physics_prior_fallback'
    assert prediction['predicted_fault'] == 'lubrication_issue'


def test_severity_band_tracks_health():
    _, healthy, good = run_pipeline(warmup=180, after=100, with_diagnostics=True)
    _, _, bad = run_pipeline(fault='misfire', warmup=150, after=180, with_diagnostics=True)
    assert good['severity_estimate']['band'] == 'INFORMATION'
    assert bad['severity_estimate']['band'] in ('CAUTION', 'WARNING')


def test_sensor_drift_is_not_mistaken_for_overheating(diagnostics_available):
    if not diagnostics_available:
        pytest.skip('trained model not present')
    _, _, drift = run_pipeline(fault='sensor_drift', warmup=180, after=260,
                               with_diagnostics=True)
    _, _, overheat = run_pipeline(fault='overheating', warmup=150, after=200,
                                  with_diagnostics=True)
    assert drift['predicted_fault'] == 'sensor_drift'
    assert overheat['predicted_fault'] == 'overheating'


def test_misfire_and_injector_fault_are_distinguished(diagnostics_available):
    """Both disturb one cylinder; the sign of its EGT deviation separates them."""
    if not diagnostics_available:
        pytest.skip('trained model not present')
    _, _, misfire = run_pipeline(fault='misfire', warmup=150, after=180, with_diagnostics=True)
    _, _, injector = run_pipeline(fault='injector_abnormality', warmup=150, after=200,
                                  with_diagnostics=True)
    assert misfire['predicted_fault'] == 'misfire'
    assert injector['predicted_fault'] == 'injector_abnormality'
