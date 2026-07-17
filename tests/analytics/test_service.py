import analytics.service as service_module
from analytics.service import AnalyticsService
from analytics.db import connect_analytics, set_meta


def test_supported_chat_returns_evidence_and_dataset_version(
    powered_analytics_db, monkeypatch
):
    service = AnalyticsService(str(powered_analytics_db))
    monkeypatch.setattr(
        service,
        "freshness",
        lambda: {
            "fresh": True,
            "source_dataset_version": "same",
            "dataset_version": "same",
        },
    )

    response = service.chat("La vision aide-t-elle à gagner en EUW ?", None, {})

    assert response.success is True
    assert response.abstained is False
    assert response.evidence
    assert response.dataset_version


def test_timeline_question_abstains_explicitly(
    powered_analytics_db, monkeypatch
):
    service = AnalyticsService(str(powered_analytics_db))
    monkeypatch.setattr(service, "freshness", lambda: {"fresh": True})

    response = service.chat("Qui gagnait à 10 minutes ?", None, {})

    assert response.abstained is True
    assert response.warnings[0]["code"] == "TIMELINE_DATA_UNAVAILABLE"


def test_region_comparison_without_metric_asks_for_one(
    powered_analytics_db, monkeypatch
):
    service = AnalyticsService(str(powered_analytics_db))
    monkeypatch.setattr(service, "freshness", lambda: {"fresh": True})

    response = service.chat("Compare EUW et KR", None, {})

    assert response.abstained is True
    assert response.warnings[0]["code"] == "UNSUPPORTED_FILTER"
    assert response.warnings[0]["details"] == {"required": "metric"}


def test_stale_mart_refuses_analysis(powered_analytics_db, monkeypatch):
    service = AnalyticsService(str(powered_analytics_db))
    monkeypatch.setattr(
        service,
        "freshness",
        lambda: {
            "fresh": False,
            "source_dataset_version": "new",
            "dataset_version": "old",
        },
    )

    response = service.chat("Comment gagne-t-on ?", None, {})

    assert response.abstained is True
    assert response.warnings[0]["code"] == "ANALYTICS_DB_STALE"


def test_missing_mart_refuses_instead_of_raising(tmp_path):
    response = AnalyticsService(str(tmp_path / "missing.db")).chat(
        "Comment gagne-t-on ?", None, {}
    )

    assert response.abstained is True
    assert response.warnings[0]["code"] == "ANALYTICS_DB_STALE"


def test_missing_mart_health_is_structured(tmp_path):
    health = AnalyticsService(str(tmp_path / "missing.db")).health()

    assert health == {"ok": False, "code": "ANALYTICS_DB_STALE"}


def test_freshness_rejects_incompatible_analytics_versions(
    powered_analytics_pair,
):
    source_db, analytics_db = powered_analytics_pair
    service = AnalyticsService(str(analytics_db), str(source_db))
    assert service.freshness()["fresh"] is True
    with connect_analytics(str(analytics_db)) as conn:
        set_meta(conn, "extractor_version", "0.9.0")

    freshness = service.freshness(force=True)
    with connect_analytics(str(analytics_db)) as conn:
        set_meta(conn, "extractor_version", "1.0.0")

    assert freshness["fresh"] is False
    assert freshness["compatibility_mismatches"]["extractor_version"] == {
        "expected": "1.0.0",
        "actual": "0.9.0",
    }


def test_freshness_snapshot_is_cached_for_short_request_bursts(
    powered_analytics_pair, monkeypatch
):
    source_db, analytics_db = powered_analytics_pair
    original = service_module.source_snapshot
    calls = 0

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(service_module, "source_snapshot", counted)
    service = AnalyticsService(str(analytics_db), str(source_db))

    assert service.freshness()["fresh"] is True
    assert service.freshness()["fresh"] is True
    assert calls == 1
