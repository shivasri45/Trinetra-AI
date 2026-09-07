"""Feature construction and the explainability layer."""
from __future__ import annotations

import numpy as np
import pytest

from backend.ml.diagnostics import Diagnostics
from backend.preprocessing.pipeline import FEATURE_NAMES, FeatureBuilder
from backend.xai.explainer import GROUPS, Explainer
from conftest import run_pipeline


# --------------------------------------------------------------------- features
def test_feature_names_are_unique_and_stable():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert FEATURE_NAMES == tuple(FEATURE_NAMES)
    assert len(FEATURE_NAMES) > 50


def test_vector_ordering_matches_the_name_list():
    features = {name: float(index) for index, name in enumerate(FEATURE_NAMES)}
    vector = FeatureBuilder.to_vector(features)
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.array_equal(vector, np.arange(len(FEATURE_NAMES), dtype=float))


def test_missing_features_become_zero_not_an_error():
    vector = FeatureBuilder.to_vector({})
    assert vector.shape == (len(FEATURE_NAMES),)
    assert not vector.any()


def test_all_declared_features_are_produced():
    _, result, _ = run_pipeline(warmup=60, after=20)
    missing = [name for name in FEATURE_NAMES if name not in result.features]
    assert not missing, missing


def test_features_are_finite_and_numeric():
    _, result, _ = run_pipeline(fault='misfire', warmup=80, after=80)
    for name, value in result.features.items():
        assert isinstance(value, float), name
        assert np.isfinite(value), f'{name} = {value}'


def test_order_features_are_dimensionless_ratios():
    """Crank-order features are referenced to the firing order so they generalise."""
    _, cruise, _ = run_pipeline(mission='normal_cruise', warmup=120, after=40)
    _, endurance, _ = run_pipeline(mission='long_endurance', warmup=120, after=40)
    for key in ('vib_order_0p5_ratio', 'vib_order_1p0_ratio'):
        assert abs(cruise.features[key] - endurance.features[key]) < 0.25, key


def test_rolling_variability_separates_oscillation_from_offset():
    """Combustion instability oscillates; overheating holds a steady offset."""
    _, unstable, _ = run_pipeline(fault='combustion_instability', warmup=120, after=120)
    _, steady, _ = run_pipeline(fault='overheating', warmup=120, after=120)
    assert unstable.features['egt_nres_std'] > steady.features['egt_nres_std']


def test_conditional_oil_pressure_feature_separates_the_two_causes():
    _, lube, _ = run_pipeline(fault='lubrication_issue', warmup=150, after=180)
    _, hot, _ = run_pipeline(fault='overheating', warmup=150, after=180)
    assert lube.features['oil_pressure_cond_nres'] < hot.features['oil_pressure_cond_nres']
    assert lube.features['oil_pressure_cond_nres'] < -3.0


def test_builder_reset_clears_the_window():
    builder = FeatureBuilder()
    _, result, _ = run_pipeline(warmup=40, after=10)
    assert not builder.nres_history
    builder.nres_history.append({})
    builder.reset()
    assert not builder.nres_history and not builder.value_history


# -------------------------------------------------------------------- explainer
def test_every_feature_belongs_to_at_least_one_group():
    explainer = Explainer(Diagnostics())
    covered: set[int] = set()
    for indices in explainer._group_index.values():
        covered.update(indices)
    uncovered = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if i not in covered]
    assert not uncovered, uncovered


def test_groups_are_non_empty():
    explainer = Explainer(Diagnostics())
    empty = [label for label, indices in explainer._group_index.items() if not indices]
    assert not empty, empty


def test_healthy_engine_explanation_states_no_deviation():
    _, result, prediction = run_pipeline(warmup=180, after=100, with_diagnostics=True)
    explanation = Explainer(Diagnostics()).explain(result.features, result.twin, prediction)
    assert 'within expected instrument scatter' in explanation['summary']
    assert explanation['sensor_plausibility'] == 'CONSISTENT'


@pytest.mark.slow
@pytest.mark.parametrize('fault,expected_group', [
    ('misfire', 'Crank-order content'),
    ('lubrication_issue', 'Oil pressure'),
    ('excessive_vibration', 'Crank-order content'),
    ('alternator_failure', 'Electrical bus'),
    ('sensor_drift', 'Cylinder head temperature'),
])
def test_attribution_points_at_the_physically_responsible_group(fault, expected_group,
                                                               diagnostics_available):
    if not diagnostics_available:
        pytest.skip('trained model not present')
    diagnostics = Diagnostics()
    explainer = Explainer(diagnostics)
    frame, result, prediction = run_pipeline(fault=fault, warmup=170, after=240,
                                             with_diagnostics=True)
    for _ in range(6):
        prediction = diagnostics.analyse(result.features, result.twin)
    explanation = explainer.explain(result.features, result.twin, prediction)

    assert explanation['attribution_method'] == 'grouped_occlusion_on_trained_model'
    groups = [item['feature'] for item in explanation['evidence']]
    assert expected_group in groups[:3], f'{fault} evidence was {groups}'


def test_attribution_shares_are_normalised():
    _, result, prediction = run_pipeline(fault='misfire', warmup=150, after=180,
                                         with_diagnostics=True)
    explanation = Explainer(Diagnostics()).explain(result.features, result.twin, prediction)
    total = sum(item['contribution'] for item in explanation['evidence'])
    assert 0 < total <= 100.5
    for item in explanation['evidence']:
        assert item['contribution'] >= 0


def test_evidence_is_auditable_against_telemetry():
    """Each channel-backed item must quote measured, expected and the deviation."""
    _, result, prediction = run_pipeline(fault='lubrication_issue', warmup=150,
                                         after=180, with_diagnostics=True)
    explanation = Explainer(Diagnostics()).explain(result.features, result.twin, prediction)
    channel_items = [item for item in explanation['evidence'] if 'channel' in item]
    assert channel_items
    for item in channel_items:
        assert {'measured', 'expected', 'residual_sigma', 'direction'} <= set(item)
        assert item['direction'] in ('above expected', 'below expected')


def test_explanation_includes_mechanism_and_runner_up():
    _, result, prediction = run_pipeline(fault='overheating', warmup=150, after=180,
                                         with_diagnostics=True)
    explanation = Explainer(Diagnostics()).explain(result.features, result.twin, prediction)
    assert len(explanation['mechanism']) > 60
    assert 'cooling' in explanation['mechanism'].lower()
    if explanation['runner_up']:
        assert explanation['runner_up']['fault'] != prediction['predicted_fault']


def test_explanation_carries_the_synthetic_disclaimer():
    _, result, prediction = run_pipeline(warmup=100, after=40, with_diagnostics=True)
    explanation = Explainer(Diagnostics()).explain(result.features, result.twin, prediction)
    assert 'not an airworthiness determination' in explanation['disclaimer'].lower()


def test_explainer_falls_back_to_residual_ranking_without_a_model():
    diagnostics = Diagnostics(model_dir='does-not-exist')
    explainer = Explainer(diagnostics)
    _, result, _ = run_pipeline(fault='lubrication_issue', warmup=150, after=180)
    for _ in range(8):
        prediction = diagnostics.analyse(result.features, result.twin)
    explanation = explainer.explain(result.features, result.twin, prediction)
    assert explanation['attribution_method'] == 'physics_rules_only'
    assert explanation['evidence']
    assert 'oil pressure' in {item['feature'] for item in explanation['evidence']}


def test_group_definitions_reference_real_channels():
    from backend.digital_twin.estimator import CHANNEL_MODELS
    for label, (_, channel) in GROUPS.items():
        if channel is not None:
            assert channel in CHANNEL_MODELS, f'{label} references unknown channel {channel}'
