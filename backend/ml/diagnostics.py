"""Fault detection and identification.

The signature of :meth:`Diagnostics.analyse` deliberately has no place for the
injected fault, so the ground truth cannot reach inference even by accident.
Everything comes from the feature vector and the twin's statistics.

Three independent detectors are fused:

1. Statistical  - mean normalised innovation squared from the Kalman twin.
   Model based, needs no training, fires on any deviation from physics.
2. Unsupervised - IsolationForest novelty trained on healthy data only. Flags
   unfamiliar behaviour, including failure modes that were never labelled.
3. Supervised   - RandomForest class probabilities for naming the fault.

Identification blends the classifier probabilities with a transparent
physics-rule prior. The prior is kept visible in the output so an engineer can
see when the learned model and the physical reasoning disagree, which is exactly
the situation that deserves human attention.
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from backend.config.settings import FAULTS, settings
from backend.preprocessing.pipeline import FEATURE_NAMES, FeatureBuilder

ML_WEIGHT = 0.75
PRIOR_WEIGHT = 0.25
CLASSIFIER_MIN_PROB = 0.55
PRIOR_MIN_PROB = 0.45

# Alarm debouncing. A single sample crossing a threshold is not an alarm; the
# condition has to persist. Throttle transients momentarily unbalance individual
# channels, so without this the transition profile generates constant nuisance
# alarms - the classic reason operators disable a monitoring system.
PERSISTENCE_WINDOW = 6
PERSISTENCE_REQUIRED = 4


def _pos(value: float) -> float:
    return max(0.0, value)


class Diagnostics:
    """Loads the trained bundle if present, otherwise degrades to physics rules."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.bundle: dict | None = None
        self.backend = 'physics_prior_fallback'
        self.trigger_history: deque[set[str]] = deque(maxlen=PERSISTENCE_WINDOW)
        path = Path(model_dir or settings.model_dir) / 'diagnostics.joblib'
        if path.exists():
            try:
                bundle = joblib.load(path)
                if list(bundle.get('feature_names', [])) != list(FEATURE_NAMES):
                    raise ValueError('feature schema changed since training')
                # Training used every core; inference is one row at a time, where
                # thread-pool setup costs far more than the work itself.
                for model in ('classifier', 'novelty'):
                    if hasattr(bundle.get(model), 'n_jobs'):
                        bundle[model].n_jobs = 1
                self.bundle = bundle
                self.backend = 'trained'
            except Exception as exc:  # pragma: no cover - defensive load path
                self.load_error = str(exc)

    # --------------------------------------------------------------- detection
    def analyse(self, features: dict[str, float], twin: dict) -> dict:
        prior = self._physics_prior(features, twin)
        prior_total = sum(prior.values()) or 1.0
        prior_probs = {k: v / prior_total for k, v in prior.items()}

        mean_nis = float(twin.get('mean_nis', 0.0))
        peak_nis = float(twin.get('peak_nis', 0.0))
        statistical_score = 1.0 - math.exp(-max(mean_nis / settings.nis_warning,
                                                peak_nis / settings.nis_peak_warning))

        novelty_score = 0.0
        novelty_raw = None
        ml_probs: dict[str, float] = {}
        if self.bundle is not None:
            vector = FeatureBuilder.to_vector(features).reshape(1, -1)
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
            scaled = self.bundle['scaler'].transform(vector)
            novelty_raw = float(-self.bundle['novelty'].score_samples(scaled)[0])
            reference = self.bundle['novelty_reference']
            spread = max(1e-6, reference['healthy_p98'] - reference['healthy_median'])
            novelty_score = 1.0 / (1.0 + math.exp(-2.0 * (novelty_raw - reference['healthy_p98']) / spread))
            probabilities = self.bundle['classifier'].predict_proba(scaled)[0]
            ml_probs = {label: float(p) for label, p in zip(self.bundle['classes'], probabilities)}

        fused = {
            fault: ML_WEIGHT * ml_probs.get(fault, 0.0) + PRIOR_WEIGHT * prior_probs.get(fault, 0.0)
            for fault in FAULTS
        } if ml_probs else dict(prior_probs)

        non_normal = {k: v for k, v in fused.items() if k != 'normal'}
        candidate = max(non_normal, key=non_normal.get) if non_normal else 'normal'
        candidate_score = non_normal.get(candidate, 0.0)

        instantaneous: set[str] = set()
        if mean_nis > settings.nis_warning:
            instantaneous.add('statistical_residual')
        if peak_nis > settings.nis_peak_warning:
            instantaneous.add('statistical_channel')
        if self.bundle is not None and novelty_raw is not None and novelty_raw > self.bundle['novelty_threshold']:
            instantaneous.add('unsupervised_novelty')
        if ml_probs and candidate_score > CLASSIFIER_MIN_PROB:
            instantaneous.add('supervised_classifier')
        elif not ml_probs and candidate_score > PRIOR_MIN_PROB:
            # With no trained bundle, the novelty and classifier detectors are both
            # unavailable, and the aggregate residual test alone misses faults
            # confined to a single channel. The physics rules then have to be able
            # to raise an alarm, or the fallback path could only ever detect the
            # loudest failures.
            instantaneous.add('physics_prior')
        # Sensor plausibility is reported for interpretation but is deliberately not
        # an alarm source on its own. A single channel can lead its coupled group
        # during a genuine fault, so using isolation alone would misattribute real
        # engine faults to instrumentation.
        plausibility_flag = bool(twin.get('sensor_validation', {}).get('suspect_channels'))

        self.trigger_history.append(instantaneous)
        counts = {name: sum(1 for past in self.trigger_history if name in past)
                  for name in set().union(*self.trigger_history)}
        triggers = sorted(name for name, count in counts.items() if count >= PERSISTENCE_REQUIRED)

        anomaly = bool(triggers)
        predicted = candidate if anomaly else 'normal'
        anomaly_score = float(np.clip(
            0.45 * statistical_score + 0.35 * novelty_score + 0.20 * candidate_score, 0.0, 1.0))

        if anomaly:
            confidence = 100.0 * float(np.clip(candidate_score / max(1e-6, sum(fused.values())), 0.0, 1.0))
            confidence = max(35.0, min(99.0, confidence))
        else:
            # Confidence that the engine is healthy comes from the classifier's own
            # normal probability; the physics prior is a coarse tie-breaker and
            # would otherwise drag a confident "normal" down for no reason.
            normal_probability = ml_probs.get('normal', prior_probs.get('normal', 0.9)) if ml_probs \
                else prior_probs.get('normal', 0.9)
            confidence = 100.0 * float(np.clip(normal_probability, 0.35, 0.99))

        return {
            'anomaly': anomaly,
            'anomaly_score': round(anomaly_score, 3),
            'predicted_fault': predicted,
            'confidence': round(confidence, 1),
            'detection_triggers': triggers,
            'instantaneous_triggers': sorted(instantaneous),
            'sensor_plausibility_flag': plausibility_flag,
            'persistence': {'window': PERSISTENCE_WINDOW, 'required': PERSISTENCE_REQUIRED,
                            'counts': counts},
            'severity_estimate': self._severity(twin, anomaly_score),
            'detectors': {
                'statistical': {'mean_nis': round(mean_nis, 2), 'score': round(statistical_score, 3),
                                'threshold': settings.nis_warning},
                'unsupervised': {'novelty_raw': None if novelty_raw is None else round(novelty_raw, 4),
                                 'threshold': None if self.bundle is None else round(self.bundle['novelty_threshold'], 4),
                                 'score': round(novelty_score, 3)},
                'supervised': {'top_class': candidate, 'probability': round(candidate_score, 3)},
            },
            'class_probabilities': {k: round(v, 4) for k, v in sorted(
                fused.items(), key=lambda kv: -kv[1])},
            'ml_probabilities': {k: round(v, 4) for k, v in sorted(
                ml_probs.items(), key=lambda kv: -kv[1])} if ml_probs else {},
            'physics_prior': {k: round(v, 4) for k, v in sorted(
                prior_probs.items(), key=lambda kv: -kv[1])},
            'agreement': self._agreement(ml_probs, prior_probs),
            'model_backend': self.backend,
        }

    # ------------------------------------------------------------------- prior
    @staticmethod
    def _physics_prior(f: dict[str, float], twin: dict) -> dict[str, float]:
        """Transparent rule scores derived from engine physics, not from labels."""
        indicators = twin.get('condition_indicators', {})
        isolation = twin.get('sensor_validation', {}).get('isolation', {})

        def ind(name: str) -> float:
            return float(indicators.get(name, {}).get('exceedance', 0.0))

        egt_low = _pos(-f.get('egt_cyl_min_dev', 0.0)) * 12.0
        egt_high = _pos(f.get('egt_cyl_max_dev', 0.0)) * 12.0

        # Every term is an *excess* over the healthy noise floor, so that a healthy
        # engine scores highest on 'normal'. Rolling-variability terms especially
        # need a dead band: healthy innovations have a standard deviation near one
        # by construction, and scoring that raw made combustion instability the
        # top hypothesis on a perfectly good engine.
        scores = {
            'normal': 0.90,
            'misfire': 2.2 * ind('misfire_index') + 1.6 * egt_low + 0.6 * _pos(-f.get('rpm_rel', 0.0)) * 8.0,
            'injector_abnormality': 1.9 * egt_high + 1.2 * abs(f.get('air_fuel_ratio_nres', 0.0)) / 3.0
                                    + 0.9 * ind('egt_spread'),
            'overheating': 1.7 * _pos(f.get('cht_nres', 0.0)) / 3.0 + 1.3 * _pos(f.get('oil_temperature_nres', 0.0)) / 3.0
                           + 0.8 * ind('cht_spread'),
            # Uses the temperature-conditioned pressure residual so that hot oil
            # alone does not read as a lubrication failure.
            'lubrication_issue': 2.0 * _pos(-f.get('oil_pressure_cond_nres', 0.0)) / 3.0
                                 + 1.1 * ind('roughness_index')
                                 + 0.5 * _pos(f.get('oil_temperature_nres', 0.0)) / 3.0,
            'excessive_vibration': 2.4 * ind('imbalance_index') + 0.8 * _pos(f.get('vibration_nres', 0.0)) / 3.0,
            # A thermal instrument reading high while its physically coupled
            # channels sit at expectation is not a thermal event.
            'sensor_drift': (1.6 * _pos(float(isolation.get('cht', 0.0)) - 1.0) / 2.0
                             + 1.2 * _pos(abs(f.get('cht_drift', 0.0)) - 1.0)
                             + 2.0 * _pos(abs(f.get('cht_nres', 0.0)) + abs(f.get('cht_drift', 0.0))
                                          - abs(f.get('oil_temperature_nres', 0.0))
                                          - abs(f.get('egt_nres', 0.0)) - 1.0) / 2.0),
            'combustion_instability': (2.0 * _pos(f.get('egt_nres_std', 0.0) - 1.0)
                                       + 1.6 * _pos(f.get('air_fuel_ratio_nres_std', 0.0) - 1.0)
                                       + 0.9 * _pos(f.get('rpm_nres_std', 0.0) - 1.0)),
            'coking_degradation': 1.6 * _pos(-f.get('fuel_flow_rel', 0.0)) * 6.0
                                  + 1.4 * _pos(-f.get('rpm_rel', 0.0)) * 6.0
                                  + 0.7 * _pos(-f.get('manifold_pressure_nres', 0.0)) / 3.0,
            'alternator_failure': 2.4 * _pos(-f.get('alternator_output_nres', 0.0)) / 3.0
                                  + 1.8 * _pos(-f.get('battery_voltage_nres', 0.0)) / 3.0,
        }
        return {k: max(0.0, v) for k, v in scores.items()}

    @staticmethod
    def _agreement(ml_probs: dict[str, float], prior_probs: dict[str, float]) -> dict:
        if not ml_probs:
            return {'state': 'prior_only', 'ml_top': None, 'prior_top': max(prior_probs, key=prior_probs.get)}
        ml_top = max(ml_probs, key=ml_probs.get)
        prior_top = max(prior_probs, key=prior_probs.get)
        return {
            'state': 'agree' if ml_top == prior_top else 'disagree',
            'ml_top': ml_top,
            'prior_top': prior_top,
            'note': ('Learned model and physical reasoning agree.' if ml_top == prior_top
                     else 'Learned model and physical reasoning disagree; engineer review advised.'),
        }

    @staticmethod
    def _severity(twin: dict, anomaly_score: float) -> dict:
        """Map health loss and residual magnitude onto an advisory severity band."""
        health = float(twin.get('health_score', 100.0))
        if health > settings.warning_health and anomaly_score < 0.5:
            band = 'INFORMATION'
        elif health > settings.critical_health:
            band = 'CAUTION'
        else:
            band = 'WARNING'
        return {'band': band, 'health_score': health, 'anomaly_score': round(anomaly_score, 3)}
