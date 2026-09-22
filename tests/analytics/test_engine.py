from analytics.config import AnalyticsConfig
from analytics.db import connect_analytics
from analytics.registry import METRICS
from analytics.types import AnalysisFilters, AnalysisRequest, Intent
from tests.analytics.factories import seed_fact_rows


def test_engine_applies_bh_to_complete_candidate_family(powered_analytics_db):
    from analytics.engine import AnalysisEngine

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
    from analytics.engine import AnalysisEngine

    engine = AnalysisEngine(str(powered_analytics_db))
    request = AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")

    first = engine.run(request, question="L'or est-il associé à la victoire ?")[0]
    second = engine.run(request, question="L'or est-il associé à la victoire ?")[0]

    assert first.evidence_id == second.evidence_id
    assert first.to_dict() == second.to_dict()


def test_different_question_wording_gets_its_own_packet(powered_analytics_db):
    from analytics.engine import AnalysisEngine

    engine = AnalysisEngine(str(powered_analytics_db))
    request = AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")

    first = engine.run(request, question="Question A sur l'or")[0]
    second = engine.run(request, question="Question B sur l'or")[0]

    assert first.evidence_id != second.evidence_id
    assert first.question != second.question


def test_cache_does_not_reuse_a_packet_from_another_seed(powered_analytics_db):
    from analytics.engine import AnalysisEngine

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
    with connect_analytics(str(powered_analytics_db), readonly=True) as conn:
        cache_rows = conn.execute(
            "SELECT evidence_id,method_version FROM evidence_cache "
            "WHERE evidence_id IN (?,?)",
            (default.evidence_id, alternate.evidence_id),
        ).fetchall()
    assert len(cache_rows) == 2
    assert len({row["method_version"] for row in cache_rows}) == 2


def test_compare_patches_returns_contrast_not_two_unrelated_rankings(
    powered_analytics_db,
):
    from analytics.engine import AnalysisEngine

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
    from analytics.engine import AnalysisEngine

    packet = AnalysisEngine(str(powered_analytics_db)).run(
        AnalysisRequest(Intent.EXPLAIN_METRIC, "atakhan"),
        question="Atakhan est-il associé à la victoire ?",
    )[0]

    assert packet.recommendation["allowed"] is False
    assert "EFFECT_NOT_ROBUST" in packet.limitations


def test_compare_regions_widens_until_each_region_is_powered(
    powered_analytics_db,
):
    from analytics.engine import AnalysisEngine

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


def test_compare_regions_applies_bh_to_every_requested_pair(
    powered_analytics_db,
):
    from analytics.engine import AnalysisEngine

    request = AnalysisRequest(
        Intent.COMPARE_REGIONS,
        "vision_score",
        AnalysisFilters(regions=("EUW", "KR", "NA")),
    )
    config = AnalyticsConfig(
        analytics_path=str(powered_analytics_db),
        bootstrap_resamples=200,
    )

    packets = AnalysisEngine(str(powered_analytics_db), config).run(
        request,
        question="Compare la vision entre EUW, KR et NA",
    )

    assert len(packets) == 3
    assert {
        frozenset(packet.sample["events_by_group"])
        for packet in packets
    } == {
        frozenset(("EUW", "KR")),
        frozenset(("EUW", "NA")),
        frozenset(("KR", "NA")),
    }
    assert all(packet.method["candidate_family_size"] == 3 for packet in packets)
    assert all(
        packet.estimate["q_value"] >= packet.estimate["p_value"]
        for packet in packets
    )


def test_compare_regions_discovers_supported_region_family_automatically(
    powered_analytics_db,
):
    from analytics.engine import AnalysisEngine

    config = AnalyticsConfig(
        analytics_path=str(powered_analytics_db),
        bootstrap_resamples=200,
    )

    packets = AnalysisEngine(str(powered_analytics_db), config).run(
        AnalysisRequest(Intent.COMPARE_REGIONS, "vision_score"),
        question="Existe-t-il des différences régionales de vision ?",
    )

    assert len(packets) == 6
    assert all(packet.method["candidate_family_size"] == 6 for packet in packets)


def test_null_regional_family_publishes_no_ranking(tmp_path):
    from analytics.engine import AnalysisEngine

    path = seed_fact_rows(
        tmp_path / "null-regions.db",
        {"16.14": 1800},
        region_shifts={"EUW": 0.0, "KR": 0.0, "NA": 0.0},
    )
    config = AnalyticsConfig(
        analytics_path=str(path),
        bootstrap_resamples=200,
    )

    packets = AnalysisEngine(str(path), config).run(
        AnalysisRequest(
            Intent.COMPARE_REGIONS,
            "vision_score",
            AnalysisFilters(regions=("EUW", "KR", "NA")),
        ),
        question="Compare la vision entre EUW, KR et NA",
    )

    assert len(packets) == 3
    assert all(packet.recommendation["allowed"] is False for packet in packets)
