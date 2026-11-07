"""Serving evidence applies only to the feature contract's history window."""
import pytest

from core.frozen_history import FrozenHistoryProvider
from core.pre_match_history import ResultOrderedHistory
from tests.serving_artifact_fixture import build_serving_artifacts
from tests.test_serving_validation_gates import load_service, match


def _prediction_match():
    value = match()
    value['prediction_time_ms'] = 100_000
    return value


@pytest.mark.parametrize('history_limit,validated', [(1, False), (10_000, True)])
def test_provider_history_window_must_match_validated_feature_policy(
    tmp_path, monkeypatch, history_limit, validated,
):
    service = load_service(
        monkeypatch, build_serving_artifacts(tmp_path),
        history_provider=FrozenHistoryProvider([], history_limit=history_limit),
    )
    result = service.predict_match(_prediction_match())

    assert result['priors_provenance']['history_window_limit'] == history_limit
    assert result['calibration']['status'] == 'artifact_verified'
    assert result['probabilities_calibrated'] is not None
    assert service.verification_status()['validated'] is validated
    assert result['selective_decision']['is_actionable'] is validated
    assert result['selective_decision']['abstained'] is not validated
    if not validated:
        assert result['selective_decision']['tier'] == 'UNVALIDATED_MODEL'
        assert result['model']['serving_validation']['status'] != 'validated'


def test_changing_provider_history_window_revokes_serving_approval(tmp_path, monkeypatch):
    provider = FrozenHistoryProvider([], history_limit=10_000)
    service = load_service(
        monkeypatch, build_serving_artifacts(tmp_path), history_provider=provider,
    )
    assert service.predict_match(_prediction_match())['selective_decision']['is_actionable']

    provider.history_limit = 1
    result = service.predict_match(_prediction_match())

    assert result['probabilities_calibrated'] is not None
    assert result['selective_decision']['tier'] == 'UNVALIDATED_MODEL'
    assert result['selective_decision']['abstained']
    assert not service.verification_status()['validated']


@pytest.mark.parametrize('history_window_limit,validated', [
    (1, False), (10_000, True), (None, True),
])
def test_supplied_history_window_must_match_validated_feature_policy(
    tmp_path, monkeypatch, history_window_limit, validated,
):
    service = load_service(monkeypatch, build_serving_artifacts(tmp_path))
    enriched = ResultOrderedHistory().enrich(
        _prediction_match(), include_timeline=False, as_of_ms=100_000,
    )
    if history_window_limit is not None:
        enriched['priors_provenance']['history_window_limit'] = history_window_limit
    result = service.predict_match(enriched)

    assert result['probabilities_calibrated'] is not None
    assert result['selective_decision']['is_actionable'] is validated
    assert result['selective_decision']['abstained'] is not validated
    if not validated:
        assert result['selective_decision']['tier'] == 'UNVALIDATED_MODEL'
        assert result['model']['serving_validation']['status'] != 'validated'
