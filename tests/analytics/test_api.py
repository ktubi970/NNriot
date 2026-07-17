from unittest.mock import MagicMock

from app import create_app


def test_chat_endpoint_preserves_response_contract(monkeypatch):
    service = MagicMock()
    service.chat.return_value.to_dict.return_value = {
        "success": True,
        "answer": "Association observée.",
        "analysis_request": {"intent": "explain_metric"},
        "evidence": [],
        "warnings": [],
        "abstained": False,
        "dataset_version": "abc",
    }
    monkeypatch.setattr(
        "app.routes.analytics.get_analytics_service", lambda: service
    )
    app = create_app()
    app.config["TESTING"] = True

    response = app.test_client().post(
        "/api/analytics/chat",
        json={
            "message": "La vision aide-t-elle ?",
            "context": None,
            "filters": {},
        },
    )

    assert response.status_code == 200
    assert response.get_json()["dataset_version"] == "abc"


def test_chat_rejects_arbitrary_context_shape(monkeypatch):
    service = MagicMock()
    service.chat.side_effect = ValueError("UNSUPPORTED_FILTER: context")
    monkeypatch.setattr(
        "app.routes.analytics.get_analytics_service", lambda: service
    )
    app = create_app()
    app.config["TESTING"] = True

    response = app.test_client().post(
        "/api/analytics/chat",
        json={
            "message": "Et en KR ?",
            "context": {"sql": "SELECT 1"},
            "filters": {},
        },
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "UNSUPPORTED_FILTER"


def test_invalid_chat_payload_is_400():
    app = create_app()
    app.config["TESTING"] = True

    response = app.test_client().post("/api/analytics/chat", json={"message": 4})

    assert response.status_code == 400
    assert response.get_json()["code"] == "INVALID_REQUEST"


def test_health_and_evidence_contracts(monkeypatch):
    service = MagicMock()
    service.health.return_value = {"ok": False, "code": "ANALYTICS_DB_STALE"}
    service.get_evidence.return_value = None
    monkeypatch.setattr(
        "app.routes.analytics.get_analytics_service", lambda: service
    )
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    health = client.get("/api/analytics/health")
    evidence = client.get("/api/analytics/evidence/unknown")

    assert health.status_code == 503
    assert evidence.status_code == 404
    assert evidence.get_json()["code"] == "EVIDENCE_NOT_FOUND"


def test_existing_status_contract_still_exists():
    app = create_app()
    app.config["TESTING"] = True

    response = app.test_client().get("/api/status")

    assert response.status_code == 200
    assert "model_loaded" in response.get_json()
