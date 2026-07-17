from analytics.service import AnalyticsService


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
