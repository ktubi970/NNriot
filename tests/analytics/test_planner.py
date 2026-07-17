import pytest

from analytics.cohorts import AnalyticsAbstention
from analytics.planner import FrenchPlanner
from analytics.types import Intent


def test_parses_metric_and_region_without_sql():
    request = FrenchPlanner().plan("La vision aide-t-elle à gagner en EUW ?")

    assert request.intent is Intent.EXPLAIN_METRIC
    assert request.metric == "vision_score"
    assert request.filters.regions == ("EUW",)


def test_region_followup_only_replaces_region():
    planner = FrenchPlanner()
    first = planner.plan("La vision aide-t-elle à gagner en EUW ?")

    second = planner.plan("Et en KR ?", context=first.to_dict())

    assert second.metric == "vision_score"
    assert second.filters.regions == ("KR",)


def test_evidence_followup_keeps_validated_request():
    planner = FrenchPlanner()
    first = planner.plan("La vision aide-t-elle à gagner en EUW ?")

    second = planner.plan("Montre les preuves", context=first.to_dict())

    assert second.intent is Intent.SHOW_EVIDENCE
    assert second.metric == "vision_score"
    assert second.filters.regions == ("EUW",)


def test_parses_explicit_patch_comparison():
    request = FrenchPlanner().plan(
        "Compare l'or entre les patchs 16.13 et 16.14"
    )

    assert request.intent is Intent.COMPARE_PATCHES
    assert request.filters.patches == ("16.13", "16.14")


def test_parses_explicit_region_comparison():
    request = FrenchPlanner().plan(
        "Existe-t-il une différence de vision entre EUW et KR ?"
    )

    assert request.intent is Intent.COMPARE_REGIONS
    assert request.metric == "vision_score"
    assert request.filters.regions == ("EUW", "KR")


@pytest.mark.parametrize(
    "message,code",
    [
        ("Qui gagnait à 10 minutes ?", "TIMELINE_DATA_UNAVAILABLE"),
        ("Décris le style de jeu de G2", "ESPORT_TEAM_IDENTITY_UNAVAILABLE"),
        ("Prouve que prendre le dragon cause la victoire", "UNSUPPORTED_INTENT"),
        ("Donne-moi une stratégie de pari certaine", "UNSUPPORTED_INTENT"),
    ],
)
def test_refuses_unsupported_claims(message, code):
    with pytest.raises(AnalyticsAbstention) as caught:
        FrenchPlanner().plan(message)

    assert caught.value.code == code


def test_context_cannot_contain_sql():
    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        FrenchPlanner().plan(
            "Et en KR ?", context={"sql": "DROP TABLE match_facts"}
        )


def test_champion_filter_only_arrives_through_structured_filters():
    request = FrenchPlanner().plan(
        "La vision aide-t-elle ?", filters={"champion": "Ahri", "role": "MIDDLE"}
    )

    assert request.filters.champion == "Ahri"
    assert request.filters.role == "MIDDLE"
