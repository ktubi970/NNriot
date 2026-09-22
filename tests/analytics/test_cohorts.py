import pytest

from analytics.cohorts import AnalyticsAbstention, CohortResolver
from analytics.types import AnalysisFilters, AnalysisRequest, Intent


@pytest.mark.integration
def test_default_selects_latest_powered_ranked_patch(cohort_db):
    cohort = CohortResolver(str(cohort_db)).resolve(
        AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")
    )

    assert cohort.queue_ids == (420,)
    assert cohort.patches == ("16.14",)
    assert cohort.widened is False
    assert cohort.matches == 500
    assert cohort.players >= 100


@pytest.mark.integration
def test_default_widens_at_most_three_patches(cohort_db_underpowered_latest):
    cohort = CohortResolver(str(cohort_db_underpowered_latest)).resolve(
        AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")
    )

    assert cohort.patches == ("16.14", "16.13")
    assert cohort.widened is True
    assert cohort.matches == 600


@pytest.mark.integration
def test_rejects_unknown_region(cohort_db):
    request = AnalysisRequest(
        Intent.EXPLAIN_METRIC,
        "gold",
        AnalysisFilters(regions=("MARS",)),
    )

    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        CohortResolver(str(cohort_db)).resolve(request)


@pytest.mark.integration
def test_abstains_when_three_patches_remain_underpowered(cohort_db_tiny):
    with pytest.raises(AnalyticsAbstention) as caught:
        CohortResolver(str(cohort_db_tiny)).resolve(
            AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")
        )

    assert caught.value.code == "INSUFFICIENT_MATCHES"
    assert caught.value.details["matches"] == 300


@pytest.mark.integration
def test_where_clause_parameterizes_player_filters(cohort_db):
    request = AnalysisRequest(
        Intent.EXPLAIN_METRIC,
        "gold",
        AnalysisFilters(champion="X' OR 1=1 --", role="TOP"),
    )
    resolver = CohortResolver(str(cohort_db))

    sql, params = resolver.where_clause(request, ("16.14",))

    assert "X' OR 1=1 --" not in sql
    assert "?" in sql
    assert "X' OR 1=1 --" in params


@pytest.mark.unit
def test_structured_filters_are_bounded_and_canonically_deduplicated():
    filters = AnalysisFilters.from_dict(
        {
            "regions": ["EUW", "EUW"],
            "patches": ["16.14", "16.14"],
            "queue_ids": [420, 420],
        }
    )

    assert filters.regions == ("EUW",)
    assert filters.patches == ("16.14",)
    assert filters.queue_ids == (420,)

    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        AnalysisFilters.from_dict({"regions": ["EUW"] * 5})
    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        AnalysisFilters.from_dict({"patches": "16.14"})
    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        AnalysisFilters.from_dict({"date_start_ms": 2**63})
    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        AnalysisFilters.from_dict(
            {"date_start_ms": 20, "date_end_ms": 10}
        )


@pytest.mark.integration
def test_direct_unbounded_filters_are_rejected_before_sql(cohort_db):
    request = AnalysisRequest(
        Intent.EXPLAIN_METRIC,
        "gold",
        AnalysisFilters(patches=tuple(f"16.{index}" for index in range(1000))),
    )

    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        CohortResolver(str(cohort_db)).where_clause(request, request.filters.patches)
