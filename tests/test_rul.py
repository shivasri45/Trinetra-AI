"""RUL estimation: trend fitting, confidence bands and TBO capping."""
from __future__ import annotations

import pytest

from backend.config.settings import SUBSYSTEMS, settings
from backend.ml.rul import MIN_SAMPLES, RulEstimator


def twin_with(health: float) -> dict:
    return {'subsystems': {name: {'health': health} for name in SUBSYSTEMS}}


def drive(estimator: RulEstimator, samples: int, start: float = 100.0,
          slope_per_hour: float = 0.0, hours_per_sample: float = 0.5,
          from_hours: float = 0.0) -> dict:
    result = {}
    for index in range(samples):
        hours = from_hours + index * hours_per_sample
        health = max(0.0, start + slope_per_hour * hours)
        result = estimator.update(hours, twin_with(health))
    return result


def test_reports_establishing_trend_before_enough_data():
    estimator = RulEstimator()
    result = drive(estimator, MIN_SAMPLES - 5)
    assert result['status'] == 'establishing_trend'
    assert result['advisory']['priority'] == 'MONITOR'
    assert 'not yet established' in result['advisory']['action']


def test_flat_health_gives_no_degradation_trend_and_tbo_limited_life():
    estimator = RulEstimator()
    result = drive(estimator, 200, start=98.0, slope_per_hour=0.0)
    for detail in result['subsystems'].values():
        assert detail['status'] in ('no_degradation_trend', 'trend_established')
    assert result['hours'] == pytest.approx(result['tbo_remaining_hours'], rel=0.02)
    assert result['advisory']['priority'] == 'SCHEDULED'


def test_declining_health_extrapolates_to_the_failure_threshold():
    """RUL must equal the health margin divided by the degradation rate."""
    estimator = RulEstimator()
    slope = -0.10
    result = drive(estimator, 320, start=100.0, slope_per_hour=slope)

    detail = result['subsystems'][result['limiting_subsystem']]
    margin = detail['health'] - settings.failure_health
    analytic = margin / abs(slope)
    assert detail['slope_per_hour'] == pytest.approx(slope, rel=0.15)
    assert detail['rul_hours'] == pytest.approx(analytic, rel=0.2)
    assert result['status'] == 'trend_established'


def test_confidence_interval_brackets_the_estimate():
    """Scatter around the trend must widen the interval, not be ignored."""
    import numpy as np
    rng = np.random.default_rng(0)
    estimator = RulEstimator()
    result = {}
    for index in range(320):
        hours = index * 0.5
        health = 100.0 - 0.08 * hours + float(rng.normal(0, 1.5))
        result = estimator.update(hours, twin_with(health))
    low, high = result['confidence_interval_hours']
    assert low <= result['hours'] <= high
    assert high > low, 'a noisy trend must carry non-zero uncertainty'


def test_a_perfectly_clean_trend_has_a_tight_interval():
    result = drive(RulEstimator(), 320, start=100.0, slope_per_hour=-0.08)
    low, high = result['confidence_interval_hours']
    assert low <= result['hours'] <= high
    assert high - low < 1.0


def test_faster_degradation_shortens_life():
    slow = drive(RulEstimator(), 320, start=100.0, slope_per_hour=-0.05)
    fast = drive(RulEstimator(), 320, start=100.0, slope_per_hour=-0.30)
    assert fast['hours'] < slow['hours']


def test_health_below_threshold_gives_zero_life_and_grounding():
    estimator = RulEstimator()
    result = drive(estimator, 320, start=settings.failure_health - 10.0, slope_per_hour=-0.2)
    assert result['hours'] == 0.0
    assert result['advisory']['priority'] == 'GROUND'


def test_rul_is_capped_by_hours_remaining_to_tbo():
    estimator = RulEstimator()
    near_tbo = settings.engine.tbo_hours - 40.0
    result = drive(estimator, 60, start=99.0, hours_per_sample=0.2, from_hours=near_tbo)
    assert result['tbo_remaining_hours'] <= 40.0
    assert result['hours'] <= result['tbo_remaining_hours'] + 1e-6


def test_limiting_subsystem_is_the_one_degrading_fastest():
    estimator = RulEstimator()
    for index in range(320):
        hours = index * 0.5
        healths = {name: 99.0 for name in SUBSYSTEMS}
        healths['lubrication'] = max(0.0, 99.0 - 0.25 * hours)
        estimator.update(hours, {'subsystems': {k: {'health': v} for k, v in healths.items()}})
    result = estimator.update(160.0, {'subsystems': {k: {'health': v} for k, v in healths.items()}})
    assert result['limiting_subsystem'] == 'lubrication'
    assert 'oil' in result['advisory']['action'].lower()


def test_advisory_priority_escalates_as_life_shortens():
    seen = {}
    for slope in (-0.02, -0.10, -0.45, -3.0):
        result = drive(RulEstimator(), 320, start=100.0, slope_per_hour=slope)
        seen[slope] = (result['hours'], result['advisory']['priority'])
    hours = [value[0] for value in seen.values()]
    assert hours == sorted(hours, reverse=True)
    assert seen[-3.0][1] in ('GROUND', 'IMMEDIATE')
    assert seen[-0.02][1] in ('SCHEDULED', 'PRIORITY')


def test_every_subsystem_is_reported():
    result = drive(RulEstimator(), 200, start=97.0, slope_per_hour=-0.05)
    assert set(result['subsystems']) == set(SUBSYSTEMS)
    for detail in result['subsystems'].values():
        assert detail['samples'] > 0
        assert 0.0 <= detail['r_squared'] <= 1.0


def test_reset_clears_the_trend():
    estimator = RulEstimator()
    drive(estimator, 200, start=100.0, slope_per_hour=-0.2)
    estimator.reset()
    result = estimator.update(0.0, twin_with(100.0))
    assert result['status'] == 'establishing_trend'


def test_output_is_labelled_as_simulated():
    result = drive(RulEstimator(), 60)
    assert 'not airworthiness data' in result['label']
    assert result['failure_threshold_health'] == settings.failure_health
    assert result['tbo_hours'] == settings.engine.tbo_hours
