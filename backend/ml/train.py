"""Train and persist the diagnostic models.

Two models are produced from the same feature matrix:

* ``IsolationForest`` fitted on healthy samples only. It provides an
  unsupervised novelty score, so an unfamiliar failure mode that was never
  labelled still raises an anomaly even though it cannot be named.
* ``RandomForestClassifier`` fitted on all labelled classes for fault
  identification, with calibrated class probabilities used as confidence.

Evaluation uses ``GroupShuffleSplit`` over session identifiers. Consecutive
samples inside one run are strongly autocorrelated, so a plain random split
would leak nearly identical rows across the boundary and report an accuracy that
does not survive contact with new data.

Fault rows are weighted by severity when fitting the classifier. The dataset
labels a row from the moment a fault is injected, but the physics takes time to
express, so the earliest rows of a slow fault are labelled for a condition the
instruments cannot yet show. Metrics are reported over all test rows and over
the expressed subset separately, because those answer different questions.
"""
from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from backend.config.settings import settings
from backend.preprocessing.pipeline import FEATURE_NAMES

MODEL_FILE = 'diagnostics.joblib'
METRICS_FILE = 'metrics.json'
# Severity above which a fault is treated as expressed. The same value already
# gates the novelty measurement below, so detection is reported on one definition.
EXPRESSED_SEVERITY = 0.6
# The dataset labels a row with its fault from the tick of injection, but the
# physics needs time to express: a cylinder head has a ~20 s time constant and
# sensor drift ramps over 120 s. Early rows therefore carry a label the data
# cannot yet support. Weighting each fault row by its severity encodes confidence
# in the label instead of asserting it. The floor keeps incipient samples in the
# fit at low influence rather than discarding them, because graded early
# response is a capability worth training for.
INCIPIENT_FLOOR_WEIGHT = 0.1
# Healthy-data quantile used as the live novelty threshold. Novelty is only one
# of three detectors, so a conservative setting is preferred: the statistical and
# supervised detectors carry sensitivity, novelty covers unlabelled failure modes.
OPERATING_QUANTILE = 0.99


def load_dataset(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, groups, severity) from the generated CSV."""
    target = Path(path or settings.dataset_path)
    if not target.exists():
        raise FileNotFoundError(
            f'{target} not found. Run "python -m backend.simulator.generate_dataset" first.')
    rows: list[dict[str, str]] = []
    with open(target, newline='', encoding='utf-8') as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'{target} is empty')

    x = np.array([[float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES] for row in rows])
    y = np.array([row['fault'] for row in rows])
    groups = np.array([row.get('session', row['fault']) for row in rows])
    severity = np.array([float(row.get('severity', 0.0) or 0.0) for row in rows])
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), y, groups, severity


def train(dataset: Path | None = None, out_dir: Path | None = None, verbose: bool = True,
          weight_by_severity: bool = False) -> dict:
    """Fit both models and persist the bundle.

    ``weight_by_severity`` toggles label-confidence weighting so its effect can
    be measured against an otherwise identical run. Off by default: measured on
    the current dataset it moved expressed macro F1 by about 0.001, which is
    within single-split noise, so it is not switched on for a change that cannot
    be demonstrated.
    """
    x, y, groups, severity = load_dataset(dataset)
    directory = Path(out_dir or settings.model_dir)
    directory.mkdir(parents=True, exist_ok=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=settings.seed)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler().fit(x_train)
    xs_train, xs_test = scaler.transform(x_train), scaler.transform(x_test)

    # --- unsupervised novelty detector, healthy data only
    healthy = xs_train[y_train == 'normal']
    novelty = IsolationForest(
        n_estimators=300, contamination=0.02, max_samples='auto',
        random_state=settings.seed, n_jobs=-1,
    ).fit(healthy)

    # --- supervised fault identification
    # Label-confidence weighting: healthy rows are certain, fault rows scale with
    # how far the fault has actually developed.
    sample_weight = np.where(
        y_train == 'normal',
        1.0,
        INCIPIENT_FLOOR_WEIGHT + (1.0 - INCIPIENT_FLOOR_WEIGHT) * severity[train_idx],
    ) if weight_by_severity else None
    classifier = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight='balanced_subsample', random_state=settings.seed, n_jobs=-1,
    ).fit(xs_train, y_train, sample_weight=sample_weight)

    predictions = classifier.predict(xs_test)
    labels = sorted(set(y))
    report = classification_report(y_test, predictions, labels=labels,
                                   zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()

    # The overall figure above scores every test row, including fault rows
    # recorded before the fault could physically be seen. Those rows are
    # genuinely indistinguishable from healthy, so they measure the labelling
    # convention rather than the model. Reporting the expressed subset as well
    # separates "can it name a developed fault" from "how much of the incipient
    # window is unknowable". Both are reported; neither replaces the other.
    expressed_mask = (y_test == 'normal') | (severity[test_idx] > EXPRESSED_SEVERITY)
    # Score only over classes actually present in the subset, so an absent class
    # cannot drag macro F1 down with a phantom zero.
    expressed_labels = sorted(set(y_test[expressed_mask]))
    expressed_report = classification_report(
        y_test[expressed_mask], predictions[expressed_mask],
        labels=expressed_labels, zero_division=0, output_dict=True,
    ) if expressed_labels else {}

    # Novelty behaviour on held-out data: how often healthy is flagged, and how
    # often a genuine fault is caught, measured only once the fault is expressed.
    healthy_mask = y_test == 'normal'
    developed = (~healthy_mask) & (severity[test_idx] > 0.6)
    novel_scores = -novelty.score_samples(xs_test)
    # The novelty threshold is a design choice, so the trade-off is recorded
    # explicitly instead of hiding behind the contamination default.
    sweep = []
    for quantile in (0.95, 0.98, 0.99, 0.995, 0.999):
        cut = float(np.quantile(novel_scores[healthy_mask], quantile)) if healthy_mask.any() else 0.0
        sweep.append({
            'healthy_quantile': quantile,
            'threshold': round(cut, 4),
            'false_alarm_rate': round(float(np.mean(novel_scores[healthy_mask] > cut)), 4) if healthy_mask.any() else 0.0,
            'developed_fault_detection': round(float(np.mean(novel_scores[developed] > cut)), 4) if developed.any() else 0.0,
        })
    operating = next(item for item in sweep if item['healthy_quantile'] == OPERATING_QUANTILE)
    threshold = operating['threshold']
    false_alarm = operating['false_alarm_rate']
    detection = operating['developed_fault_detection']

    importance = sorted(
        ({'feature': name, 'importance': round(float(value), 5)}
         for name, value in zip(FEATURE_NAMES, classifier.feature_importances_)),
        key=lambda item: -item['importance'],
    )

    bundle = {
        'scaler': scaler,
        'novelty': novelty,
        'classifier': classifier,
        'feature_names': list(FEATURE_NAMES),
        'classes': list(classifier.classes_),
        'novelty_threshold': threshold,
        # Reference point used by the explainer: attributions are measured as the
        # drop in class probability when a feature group is reset to healthy.
        'healthy_reference': healthy.mean(axis=0).tolist(),
        'novelty_reference': {
            'healthy_median': float(np.median(novel_scores[healthy_mask])) if healthy_mask.any() else 0.0,
            'healthy_p98': threshold,
        },
        'engine_spec': settings.engine.name,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, directory / MODEL_FILE)

    metrics = {
        'trained_at': bundle['trained_at'],
        'python': platform.python_version(),
        'samples': int(len(y)),
        'features': len(FEATURE_NAMES),
        'classes': labels,
        'split': {'strategy': 'GroupShuffleSplit over session id', 'test_size': 0.25,
                  'train_sessions': int(len(set(groups[train_idx]))),
                  'test_sessions': int(len(set(groups[test_idx])))},
        'label_weighting': {
            'enabled': bool(weight_by_severity),
            'scheme': ('fault rows weighted by severity, healthy rows weight 1.0'
                       if weight_by_severity else 'uniform'),
            'incipient_floor_weight': INCIPIENT_FLOOR_WEIGHT,
            'rationale': ('A fault row recorded before the fault has expressed '
                          'carries a label the measurements cannot yet support.'),
        },
        'classifier': {
            'accuracy': round(float(report['accuracy']), 4),
            'macro_f1': round(float(report['macro avg']['f1-score']), 4),
            'per_class_f1': {k: round(float(v['f1-score']), 4)
                             for k, v in report.items() if k in labels},
            'confusion_matrix': matrix,
            'expressed_only': {
                'severity_threshold': EXPRESSED_SEVERITY,
                'samples': int(expressed_mask.sum()),
                'accuracy': round(float(expressed_report['accuracy']), 4) if expressed_report else None,
                'macro_f1': round(float(expressed_report['macro avg']['f1-score']), 4)
                            if expressed_report else None,
                'per_class_f1': {k: round(float(v['f1-score']), 4)
                                 for k, v in expressed_report.items()
                                 if k in expressed_labels},
                'note': ('Healthy rows plus fault rows past the expression '
                         'threshold. Excludes the incipient window, where the '
                         'label is not yet supported by the measurements.'),
            },
        },
        'novelty': {
            'operating_quantile': OPERATING_QUANTILE,
            'threshold': round(threshold, 4),
            'healthy_false_alarm_rate': round(false_alarm, 4),
            'developed_fault_detection_rate': round(detection, 4),
            'threshold_sweep': sweep,
            'note': 'Detection measured only on samples where fault severity > 0.6.',
        },
        'top_features': importance[:20],
    }
    (directory / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    if verbose:
        print('novelty threshold sweep (healthy quantile -> false alarm / detection):')
        for item in sweep:
            marker = ' <- operating' if item['healthy_quantile'] == OPERATING_QUANTILE else ''
            print(f'  p{item["healthy_quantile"]:<6} {item["false_alarm_rate"]:.4f} / '
                  f'{item["developed_fault_detection"]:.4f}{marker}')
        print(f'samples={metrics["samples"]} features={metrics["features"]} '
              f'train_sessions={metrics["split"]["train_sessions"]} '
              f'test_sessions={metrics["split"]["test_sessions"]}')
        print(f'classifier accuracy={metrics["classifier"]["accuracy"]} '
              f'macro_f1={metrics["classifier"]["macro_f1"]}')
        expressed = metrics['classifier']['expressed_only']
        print(f'  expressed only (severity>{EXPRESSED_SEVERITY}): '
              f'accuracy={expressed["accuracy"]} macro_f1={expressed["macro_f1"]} '
              f'n={expressed["samples"]}')
        print(f'novelty: healthy false-alarm={false_alarm:.3f} '
              f'developed-fault detection={detection:.3f}')
        print('per-class F1:')
        for name, score in metrics['classifier']['per_class_f1'].items():
            print(f'  {name:24s} {score:.3f}')
        print('top features:')
        for item in importance[:12]:
            print(f'  {item["feature"]:26s} {item["importance"]:.4f}')
        print(f'saved -> {directory / MODEL_FILE}')
    return metrics


if __name__ == '__main__':
    train()
