"""Serving approval must come from verified, configuration-bound evidence."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.calibration import sha256_file
from core.prediction_service import PredictionService
from core.serving_contract import ContractError, load_model_manifest
from tests.serving_artifact_fixture import build_serving_artifacts, write_json


class FixedWinnerModel:
    inputs = [SimpleNamespace(name=k, shape=(None, 5, 14)) for k in ('team_a', 'team_b')]
    output = {'winner': SimpleNamespace(shape=(None, 2))}

    def predict(self, inputs, verbose=0):
        return self(inputs)

    def __call__(self, inputs, training=False):
        return {'winner': np.tile([.1, .9], (len(inputs['team_a']), 1))}


def match():
    return {key: [{'champion': 'Ahri', 'role': role}
                  for role in ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')]
            for key in ('team_a', 'team_b')}


def load_service(monkeypatch, artifacts, **kwargs):
    # Deserialization is slow/external to these gates; verification remains real.
    with monkeypatch.context() as context:
        context.setattr('tensorflow.keras.models.load_model', lambda *a, **k: FixedWinnerModel())
        return PredictionService(checkpoint_path=artifacts['checkpoint'],
                                 manifest_path=artifacts['manifest'], **kwargs)


def alter_report(artifacts, change):
    report = json.loads(artifacts['report'].read_text())
    change(report)
    write_json(artifacts['report'], report)
    manifest = json.loads(artifacts['manifest'].read_text())
    manifest['serving_validation']['independent_evaluation_report']['sha256'] = sha256_file(artifacts['report'])
    write_json(artifacts['manifest'], manifest)


@pytest.mark.parametrize('key,value', [('_temperature', 2.), ('_verified_decision_policy', {}),
                                      ('_checkpoint_path', 'fake'), ('_calibration_sha256', 'fake')])
def test_manifest_rejects_caller_supplied_internal_state(tmp_path, key, value):
    artifacts = build_serving_artifacts(tmp_path, calibrated=False, validated=False)
    manifest = json.loads(artifacts['manifest'].read_text())
    manifest[key] = value
    write_json(artifacts['manifest'], manifest)
    with pytest.raises(ContractError, match='[Ii]nternal|[Rr]eserved'):
        load_model_manifest(artifacts['manifest'])


def test_unverified_in_memory_model_cannot_manufacture_calibration(monkeypatch):
    service = PredictionService()
    service._model = FixedWinnerModel()
    service._is_calibrated = True
    result = service.predict_match(match())
    assert result['probabilities_calibrated'] is None
    assert result['calibration']['status'] == 'unavailable'
    assert result['selective_decision']['abstained']
    assert result['winner']['predicted'] == 'abstain'


def test_valid_bound_evidence_can_authorize_a_prediction(tmp_path, monkeypatch):
    service = load_service(monkeypatch, build_serving_artifacts(tmp_path, temperature=2.))
    result = service.predict_match(match())
    assert result['probabilities_raw']['team_b'] == pytest.approx(.9)
    assert result['probabilities_calibrated']['team_b'] == pytest.approx(.75)
    assert result['selective_decision']['is_actionable']
    assert not result['selective_decision']['abstained']
    assert result['winner']['predicted'] == 'team_b'


@pytest.mark.parametrize('calibrated,validated,tier', [
    (False, False, 'UNCALIBRATED'), (True, False, 'UNVALIDATED_MODEL'),
])
def test_missing_artifact_guarantee_forces_abstention(tmp_path, monkeypatch, calibrated, validated, tier):
    service = load_service(monkeypatch, build_serving_artifacts(tmp_path, calibrated=calibrated, validated=validated))
    result = service.predict_match(match())
    assert result['selective_decision']['tier'] == tier
    assert result['selective_decision']['abstained']
    assert not result['selective_decision']['is_actionable']
    assert (result['probabilities_calibrated'] is not None) is calibrated


def test_removing_manifest_from_calibrated_service_forces_abstention(tmp_path, monkeypatch):
    service = load_service(monkeypatch, build_serving_artifacts(tmp_path))
    service.manifest = None
    result = service.predict_match(match())
    assert result['selective_decision']['tier'] == 'UNVALIDATED_MODEL'
    assert result['selective_decision']['abstained']


def test_three_field_validation_declaration_is_not_evidence(tmp_path):
    artifacts = build_serving_artifacts(tmp_path)
    alter_report(artifacts, lambda report: [report.pop(k) for k in list(report)
                                          if k not in ('status', 'checkpoint_sha256', 'feature_contract')])
    with pytest.raises(ContractError):
        load_model_manifest(artifacts['manifest'])


@pytest.mark.parametrize('field', ['calibration_sha256', 'split_manifest_sha256', 'dataset_sha256',
                                  'runtime_feature_sources', 'decision_policy'])
@pytest.mark.parametrize('change', ['missing', 'different'])
def test_serving_evidence_must_bind_the_evaluated_configuration(tmp_path, field, change):
    artifacts = build_serving_artifacts(tmp_path)
    def alter(report):
        if change == 'missing':
            report.pop(field)
        elif field == 'decision_policy':
            report[field]['action_threshold'] = .01
        elif field == 'runtime_feature_sources':
            report[field][next(iter(report[field]))] = '0' * 64
        else:
            report[field] = '0' * 64
    alter_report(artifacts, alter)
    with pytest.raises(ContractError):
        load_model_manifest(artifacts['manifest'])


@pytest.mark.parametrize('case', ['training_id', 'empty', 'wrong_count', 'missing_metrics',
                                 'calibration_partition', 'invalid_metric'])
def test_serving_evidence_requires_the_held_out_test_cohort(tmp_path, case):
    artifacts = build_serving_artifacts(tmp_path)
    def change(report):
        evaluation = report['evaluation']
        if case == 'training_id':
            evaluation['match_ids'][0] = 'train_0'
        elif case == 'empty':
            evaluation['match_ids'] = []
            evaluation['count'] = 0
        elif case == 'wrong_count':
            evaluation['count'] = 24
        elif case == 'calibration_partition':
            evaluation['partition'] = 'validation'
        elif case == 'invalid_metric':
            evaluation['metrics']['log_loss'] = -1.
        else:
            evaluation.pop('metrics')
    alter_report(artifacts, change)
    with pytest.raises(ContractError):
        load_model_manifest(artifacts['manifest'])


def test_changed_calibration_cannot_reuse_old_serving_approval(tmp_path):
    artifacts = build_serving_artifacts(tmp_path)
    calibration = json.loads(artifacts['calibration'].read_text())
    calibration['temperature'] = calibration['fit']['temperature'] = 2.
    write_json(artifacts['calibration'], calibration)
    manifest = json.loads(artifacts['manifest'].read_text())
    manifest['calibration']['sha256'] = sha256_file(artifacts['calibration'])
    write_json(artifacts['manifest'], manifest)
    with pytest.raises(ContractError):
        load_model_manifest(artifacts['manifest'])


@pytest.mark.parametrize('change', ['request_threshold', 'service_threshold', 'high_threshold'])
def test_changed_effective_policy_forces_abstention(tmp_path, monkeypatch, change):
    service = load_service(monkeypatch, build_serving_artifacts(tmp_path))
    kwargs = {}
    if change == 'request_threshold':
        kwargs['action_threshold'] = 0.
    elif change == 'service_threshold':
        service.action_threshold = 0.
    else:
        service.high_threshold = .1
    result = service.predict_match(match(), **kwargs)
    assert result['probabilities_calibrated'] is not None
    assert result['selective_decision']['tier'] == 'UNVALIDATED_MODEL'
    assert result['selective_decision']['abstained']
    assert result['model']['serving_validation']['status'] != 'validated'


def test_changed_in_memory_temperature_cannot_claim_verified_calibration(tmp_path, monkeypatch):
    service = load_service(monkeypatch, build_serving_artifacts(tmp_path))
    service._temperature = 2.
    result = service.predict_match(match())
    assert result['probabilities_calibrated'] is None
    assert result['selective_decision']['tier'] == 'UNCALIBRATED'
    assert result['selective_decision']['abstained']


def test_mutated_manifest_cannot_retain_verified_serving_status(tmp_path, monkeypatch):
    service = load_service(monkeypatch, build_serving_artifacts(tmp_path))
    service.manifest['serving_validation']['independent_evaluation_report']['sha256'] = '0' * 64
    result = service.predict_match(match())
    assert result['selective_decision']['tier'] == 'UNVALIDATED_MODEL'
    assert result['selective_decision']['abstained']


def test_explicit_unchanged_threshold_preserves_verified_serving_status(tmp_path, monkeypatch):
    service = load_service(monkeypatch, build_serving_artifacts(tmp_path))
    result = service.predict_match(match(), action_threshold=.04)
    assert result['model']['serving_validation']['status'] == 'validated'
    assert result['selective_decision']['is_actionable']
