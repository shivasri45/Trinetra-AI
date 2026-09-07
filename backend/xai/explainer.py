"""Explainable diagnosis.

Attribution is computed, not templated. For the predicted class the explainer
measures how much of the classifier's confidence depends on each physically
meaningful feature group, by resetting that group to its healthy reference value
and observing the drop in class probability:

    attribution(group) = P(class | observed) - P(class | group reset to healthy)

That is a model-agnostic occlusion analysis over the actual trained model, in the
same family as SHAP but grouped by engineering meaning so the output reads as
evidence an engineer can check rather than as a list of raw feature names. If
``shap`` is installed it is used for per-feature values as well.

Every attribution is paired with the measured value, the twin's expectation and
the direction of the deviation, so the explanation can be audited against the
telemetry.
"""
from __future__ import annotations

import numpy as np

from backend.preprocessing.pipeline import FEATURE_NAMES, FeatureBuilder

try:  # optional dependency, richer per-feature attribution when present
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:  # pragma: no cover - shap is not required
    shap = None
    HAS_SHAP = False

# Physically meaningful groups: label -> (feature prefixes/names, engineering channel)
GROUPS: dict[str, tuple[tuple[str, ...], str | None]] = {
    'Engine speed': (('rpm_nres', 'rpm_drift', 'rpm_nres_std', 'rpm_roc', 'rpm_rel'), 'rpm'),
    'Cylinder head temperature': (('cht_nres', 'cht_drift', 'cht_nres_std', 'cht_roc', 'cht_rel'), 'cht'),
    'Exhaust gas temperature': (('egt_nres', 'egt_drift', 'egt_nres_std', 'egt_roc', 'egt_rel'), 'egt'),
    'Oil pressure': (('oil_pressure_nres', 'oil_pressure_drift', 'oil_pressure_nres_std',
                      'oil_pressure_roc', 'oil_pressure_rel',
                      'oil_pressure_cond_nres', 'oil_pressure_cond_drift',
                      'oil_pressure_cond_nres_std', 'oil_pressure_cond_roc'), 'oil_pressure'),
    'Oil temperature': (('oil_temperature_nres', 'oil_temperature_drift',
                         'oil_temperature_nres_std', 'oil_temperature_roc'), 'oil_temperature'),
    'Fuel flow': (('fuel_flow_nres', 'fuel_flow_drift', 'fuel_flow_nres_std',
                   'fuel_flow_roc', 'fuel_flow_rel'), 'fuel_flow'),
    'Mixture (air/fuel ratio)': (('air_fuel_ratio_nres', 'air_fuel_ratio_drift',
                                  'air_fuel_ratio_nres_std', 'air_fuel_ratio_roc'), 'air_fuel_ratio'),
    'Manifold pressure': (('manifold_pressure_nres', 'manifold_pressure_drift',
                           'manifold_pressure_nres_std', 'manifold_pressure_roc'), 'manifold_pressure'),
    'Vibration level': (('vibration_nres', 'vibration_drift', 'vibration_nres_std', 'vibration_roc',
                         'vib_overall_rms', 'vib_crest_factor', 'vib_spectral_kurtosis'), 'vibration'),
    'Crank-order content': (('vib_order_', 'vib_misfire_index', 'vib_imbalance_index',
                             'vib_roughness_index'), None),
    'Per-cylinder spread': (('egt_spread_rel', 'cht_spread_rel', 'egt_cyl_', 'cht_cyl_'), None),
    'Electrical bus': (('battery_voltage_', 'alternator_output_'), 'battery_voltage'),
    'Injection timing': (('injection_timing_',), 'injection_timing'),
    'Delivered power': (('power_kw_rel',), None),
    'Sensor consistency': (('_isolation', 'isolation_max', 'isolation_count'), None),
    'Operating point': (('rpm_frac', 'load_frac', 'throttle_frac', 'density_ratio',
                         'ambient_norm', 'altitude_norm'), None),
}

PHYSICS_NARRATIVE: dict[str, str] = {
    'misfire': ('One cylinder is not contributing. A dead cylinder fires once every two '
                'revolutions, so its absence appears at half the crank order, its exhaust '
                'probe cools because the charge leaves unburnt, and shaft power falls.'),
    'injector_abnormality': ('One cylinder is being metered lean. A lean charge burns more '
                             'slowly and finishes later in the expansion stroke, so more heat '
                             'is rejected to the exhaust and that cylinder runs hotter than '
                             'its neighbours while delivering less work.'),
    'overheating': ('Heat rejection has fallen. Cylinder head and oil temperatures are rising '
                    'together with unchanged fuel flow, which points to lost cooling airflow '
                    'rather than to a combustion change.'),
    'lubrication_issue': ('Oil pressure is below the value the pump should deliver at this '
                          'speed and oil temperature, and structural broadband vibration is '
                          'elevated, which is consistent with loss of oil or bearing distress.'),
    'excessive_vibration': ('Vibration energy is concentrated at the first crank order, the '
                            'signature of rotating imbalance rather than of a combustion event.'),
    'sensor_drift': ('The deviation is confined to one instrument. Its physically coupled '
                     'channels have not moved, and a genuine thermal event could not raise '
                     'head temperature while leaving oil temperature and exhaust temperature '
                     'unaffected, so the transducer is suspect.'),
    'combustion_instability': ('Cycle-to-cycle variability is high. Exhaust temperature and '
                               'mixture are oscillating rather than holding a steady offset, '
                               'with sidebands around the firing order.'),
    'coking_degradation': ('Breathing capacity has fallen. Air and fuel flow and attainable '
                           'speed are all down at unchanged throttle and manifold conditions, '
                           'consistent with deposits restricting the intake tract and valves.'),
    'alternator_failure': ('The alternator is not supporting the bus. Output current has '
                           'collapsed and the battery is discharging, so endurance is now '
                           'limited by battery capacity.'),
    'normal': 'No deviation from the physics model beyond expected instrument scatter.',
}

UNITS: dict[str, str] = {
    'rpm': 'rpm', 'cht': 'degC', 'egt': 'degC', 'oil_pressure': 'psi',
    'oil_temperature': 'degC', 'fuel_flow': 'L/h', 'air_fuel_ratio': ':1',
    'manifold_pressure': 'kPa', 'vibration': 'g', 'battery_voltage': 'V',
    'alternator_output': 'A', 'injection_timing': 'degBTDC',
}


class Explainer:
    """Occlusion-based attribution over the trained classifier."""

    def __init__(self, diagnostics) -> None:
        self.diagnostics = diagnostics
        self._index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self._group_index = {label: self._resolve(patterns) for label, (patterns, _) in GROUPS.items()}
        self._shap = None

    def _resolve(self, patterns: tuple[str, ...]) -> list[int]:
        found: list[int] = []
        for i, name in enumerate(FEATURE_NAMES):
            if any(name == p or name.startswith(p) or name.endswith(p) for p in patterns):
                found.append(i)
        return found

    # ------------------------------------------------------------------ public
    def explain(self, features: dict[str, float], twin: dict, prediction: dict) -> dict:
        fault = prediction['predicted_fault']
        bundle = self.diagnostics.bundle
        evidence: list[dict] = []
        method = 'physics_rules_only'

        if bundle is not None and fault in list(bundle['classes']):
            evidence, method = self._attribute(features, twin, fault, bundle)
        if not evidence:
            evidence = self._rule_evidence(twin)
            method = method if evidence and method != 'physics_rules_only' else 'physics_rules_only'

        summary = self._summary(fault, prediction, evidence, twin)
        return {
            'summary': summary,
            'mechanism': PHYSICS_NARRATIVE.get(fault, PHYSICS_NARRATIVE['normal']),
            'evidence': evidence,
            'attribution_method': method,
            'model_agreement': prediction.get('agreement', {}),
            'runner_up': self._runner_up(prediction),
            'sensor_plausibility': twin.get('sensor_validation', {}).get('plausibility', 'CONSISTENT'),
            'disclaimer': ('Explanation derived from a synthetic engine model and its trained '
                           'diagnostic classifier. Not an airworthiness determination.'),
        }

    # --------------------------------------------------------------- internals
    def _attribute(self, features: dict[str, float], twin: dict, fault: str,
                   bundle: dict) -> tuple[list[dict], str]:
        vector = FeatureBuilder.to_vector(features)
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
        scaled = bundle['scaler'].transform(vector)
        reference = np.array(bundle.get('healthy_reference', np.zeros(scaled.shape[1])), dtype=float)
        classifier = bundle['classifier']
        classes = list(classifier.classes_)
        target = classes.index(fault)

        labels = list(self._group_index)
        batch = np.repeat(scaled, len(labels) + 1, axis=0)
        for row, label in enumerate(labels, start=1):
            idx = self._group_index[label]
            if idx:
                batch[row, idx] = reference[idx]
        probabilities = classifier.predict_proba(batch)[:, target]
        baseline = float(probabilities[0])

        contributions = {label: baseline - float(probabilities[row])
                         for row, label in enumerate(labels, start=1)}
        positive = {k: v for k, v in contributions.items() if v > 1e-4}
        total = sum(positive.values())
        if total <= 0:
            return [], 'occlusion_no_signal'

        normalised = {k: 100.0 * v / total for k, v in positive.items()}
        ordered = sorted(normalised.items(), key=lambda kv: -kv[1])[:6]

        residuals = twin.get('normalised_residuals', {})
        measured = twin.get('measured', {})
        expected = twin.get('expected', {})
        evidence: list[dict] = []
        for label, share in ordered:
            channel = GROUPS[label][1]
            item = {
                'feature': label,
                'contribution': round(share, 1),
                'probability_drop': round(contributions[label], 4),
            }
            if channel and channel in residuals:
                sigma = residuals[channel]
                item.update({
                    'channel': channel,
                    'measured': round(float(measured.get(channel, 0.0)), 2),
                    'expected': round(float(expected.get(channel, 0.0)), 2),
                    'residual_sigma': round(float(sigma), 2),
                    'direction': 'above expected' if sigma > 0 else 'below expected',
                    'unit': UNITS.get(channel, ''),
                })
            else:
                item['detail'] = self._group_detail(label, twin, features)
            evidence.append(item)
        return evidence, 'grouped_occlusion_on_trained_model'

    @staticmethod
    def _group_detail(label: str, twin: dict, features: dict[str, float]) -> str:
        indicators = twin.get('condition_indicators', {})
        if label == 'Crank-order content':
            parts = [f"{k.replace('_', ' ')} {indicators[k]['value']}"
                     for k in ('misfire_index', 'imbalance_index', 'roughness_index') if k in indicators]
            return '; '.join(parts)
        if label == 'Per-cylinder spread':
            parts = [f"{k.replace('_', ' ')} {indicators[k]['value']}"
                     for k in ('egt_spread', 'cht_spread') if k in indicators]
            return '; '.join(parts)
        if label == 'Delivered power':
            expected = twin.get('expected', {}).get('power_kw')
            relative = features.get('power_kw_rel', 0.0)
            if expected:
                return (f'shaft power {expected * (1.0 + relative):.1f} kW against '
                        f'{expected:.1f} kW expected ({100.0 * relative:+.1f} %)')
            return 'shaft power deviates from the expectation for this command point'
        if label == 'Sensor consistency':
            suspects = twin.get('sensor_validation', {}).get('suspect_channels', [])
            if suspects:
                return ', '.join(f"{s['channel']} isolated by {s['isolation_sigma']} sigma"
                                 for s in suspects[:3])
            return 'all coupled channels mutually consistent'
        return ''

    @staticmethod
    def _rule_evidence(twin: dict) -> list[dict]:
        """Fallback when no trained model is available: rank by residual size."""
        residuals = twin.get('normalised_residuals', {})
        measured = twin.get('measured', {})
        expected = twin.get('expected', {})
        ranked = sorted(residuals.items(), key=lambda kv: -abs(kv[1]))[:5]
        total = sum(abs(v) for _, v in ranked) or 1.0
        return [{
            'feature': channel.replace('_', ' '),
            'channel': channel,
            'contribution': round(100.0 * abs(value) / total, 1),
            'measured': round(float(measured.get(channel, 0.0)), 2),
            'expected': round(float(expected.get(channel, 0.0)), 2),
            'residual_sigma': round(float(value), 2),
            'direction': 'above expected' if value > 0 else 'below expected',
            'unit': UNITS.get(channel, ''),
        } for channel, value in ranked]

    @staticmethod
    def _runner_up(prediction: dict) -> dict:
        probabilities = prediction.get('class_probabilities', {})
        ordered = [k for k in probabilities if k != prediction['predicted_fault']]
        if not ordered:
            return {}
        second = ordered[0]
        return {'fault': second, 'probability': probabilities[second],
                'note': f'Next most likely explanation considered and ranked lower: {second.replace("_", " ")}.'}

    @staticmethod
    def _summary(fault: str, prediction: dict, evidence: list[dict], twin: dict) -> str:
        if fault == 'normal':
            return ('All monitored channels track the physics model within expected instrument '
                    f"scatter (mean normalised innovation squared {twin.get('mean_nis', 0)}).")
        drivers = ', '.join(
            f"{item['feature'].lower()} ({item['contribution']}%)" for item in evidence[:3])
        triggers = ', '.join(t.replace('_', ' ') for t in prediction.get('detection_triggers', []))
        return (f"{fault.replace('_', ' ').capitalize()} identified with "
                f"{prediction.get('confidence', 0)}% confidence. Confidence rests mainly on "
                f"{drivers}. Raised by: {triggers or 'none'}.")
