import pytest

from analytics.registry import FeatureClass, MetricKind, get_metric
from analytics.types import AnalysisFilters, AnalysisRequest, Intent


def test_registry_exposes_named_final_game_metric():
    gold = get_metric("gold")

    assert gold.column == "gold"
    assert gold.kind is MetricKind.CONTINUOUS
    assert gold.feature_class is FeatureClass.FINAL_GAME_RETROSPECTIVE
    assert gold.unit == "or"


def test_registry_rejects_target_field():
    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        get_metric("win")


def test_request_round_trip_is_canonical():
    request = AnalysisRequest(
        intent=Intent.EXPLAIN_METRIC,
        metric="vision_score",
        filters=AnalysisFilters(regions=("EUW",), queue_ids=(420,)),
    )

    assert AnalysisRequest.from_dict(request.to_dict()) == request
