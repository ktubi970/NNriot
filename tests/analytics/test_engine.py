from analytics.engine import AnalysisEngine
from analytics.config import AnalyticsConfig
from analytics.registry import METRICS
from analytics.types import AnalysisFilters, AnalysisRequest, Intent


def test_engine_applies_bh_to_complete_candidate_family(powered_analytics_db):
    packets = AnalysisEngine(str(powered_analytics_db)).run(
        AnalysisRequest(Intent.RANK_CONDITIONS),
        question="Comment gagne-t-on ?",
    )

    assert len(packets) == len(METRICS) == 19
    assert all(packet.estimate["q_value"] is not None for packet in packets)
    assert all(
        packet.estimate["q_value"] >= packet.estimate["p_value"]
        for packet in packets
    )
    assert all(packet.provenance["dataset_version"] for packet in packets)
    assert {packet.estimate["metric"] for packet in packets} == set(METRICS)
    assert all(packet.method["candidate_family_size"] == 19 for packet in packets)
    gold = next(packet for packet in packets if packet.estimate["metric"] == "gold")
    assert gold.sample["matches"] == 700
    assert len(gold.evidence_id) == 64
    assert len(gold.provenance["source_match_ids_hash"]) == 64


def test_same_dataset_request_and_seed_reproduce_evidence_id(powered_analytics_db):
    engine = AnalysisEngine(str(powered_analytics_db))
    request = AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")

    first = engine.run(request, question="L'or est-il associé à la victoire ?")[0]
    second = engine.run(request, question="L'or est-il associé à la victoire ?")[0]

    assert first.evidence_id == second.evidence_id
    assert first.to_dict() == second.to_dict()


def test_different_question_wording_gets_its_own_packet(powered_analytics_db):
    engine = AnalysisEngine(str(powered_analytics_db))
    request = AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")

    first = engine.run(request, question="Question A sur l'or")[0]
    second = engine.run(request, question="Question B sur l'or")[0]

    assert first.evidence_id != second.evidence_id
    assert first.question != second.question


def test_cache_does_not_reuse_a_packet_from_another_seed(powered_analytics_db):
    request = AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")
    default = AnalysisEngine(str(powered_analytics_db)).run(
        request, question="Question réservée au test de graine"
    )[0]
    alternate_config = AnalyticsConfig(
        analytics_path=str(powered_analytics_db),
        seed=123,
        bootstrap_resamples=200,
    )

    alternate = AnalysisEngine(
        str(powered_analytics_db), alternate_config
    ).run(request, question="Question réservée au test de graine")[0]

    assert alternate.method["seed"] == 123
    assert alternate.evidence_id != default.evidence_id


def test_compare_patches_returns_contrast_not_two_unrelated_rankings(
    powered_analytics_db,
):
    request = AnalysisRequest(
        Intent.COMPARE_PATCHES,
        "gold",
        AnalysisFilters(patches=("16.13", "16.14")),
    )

    packet = AnalysisEngine(str(powered_analytics_db)).run(
        request,
        question="L'effet de l'or a-t-il changé ?",
    )[0]

    assert packet.estimate["metric"] == "gold_patch_contrast"
    assert packet.sample["events_by_group"]["16.13"] >= 100
    assert packet.sample["events_by_group"]["16.14"] >= 100


def test_null_metric_remains_descriptive(powered_analytics_db):
    packet = AnalysisEngine(str(powered_analytics_db)).run(
        AnalysisRequest(Intent.EXPLAIN_METRIC, "atakhan"),
        question="Atakhan est-il associé à la victoire ?",
    )[0]

    assert packet.recommendation["allowed"] is False
    assert "EFFECT_NOT_ROBUST" in packet.limitations


def test_compare_regions_widens_until_each_region_is_powered(
    powered_analytics_db,
):
    request = AnalysisRequest(
        Intent.COMPARE_REGIONS,
        "gold",
        AnalysisFilters(regions=("EUW", "KR")),
    )

    packet = AnalysisEngine(str(powered_analytics_db)).run(
        request,
        question="EUW et KR gagnent-elles de la même manière ?",
    )[0]

    assert packet.estimate["metric"] == "gold_region_contrast"
    assert packet.cohort["patches"] == ["16.14", "16.13", "16.12"]
    assert packet.cohort["widened"] is True
    assert min(packet.sample["events_by_group"].values()) >= 500
    assert packet.recommendation["allowed"] is True
    assert set(packet.robustness["regional_effects"]) == {"EUW", "KR"}
