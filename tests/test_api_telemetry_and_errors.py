import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app import app as flask_app
from app.services.system_service import SystemService
from core import database


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "telemetry.db")
    database.init_db(test_db)
    database.save_player("p1", "TestPlayer", "EUW", db_path=test_db)
    flask_app.config["TESTING"] = True
    flask_app.config["NNRIOT_DB_PATH"] = test_db
    return flask_app.test_client()


def test_telemetry_endpoints_are_consistent(client):
    r_health = client.get("/api/db-health")
    r_stats = client.get("/api/db/stats")
    r_metrics = client.get("/api/monitor/metrics")

    assert r_health.status_code == 200
    assert r_stats.status_code == 200
    assert r_metrics.status_code == 200

    d_health = r_health.get_json()
    d_stats = r_stats.get_json()
    d_metrics = r_metrics.get_json()

    assert d_health["players"] == 1
    assert d_stats["total_players"] == 1
    assert d_metrics["players"] == 1

    # Verify key backward-compatibility guarantees
    assert d_health["ok"] is True
    assert "matches" in d_health
    assert "training_records" in d_health

    assert d_stats["success"] is True
    assert "total_matches" in d_stats
    assert "total_training" in d_stats
    assert "crawl_queue_pending" in d_stats
    assert "db_size_mb" in d_stats
    assert "riot_api_configured" in d_stats

    assert "regions" in d_metrics
    assert "training_config" in d_metrics
    assert "riot_api" in d_metrics
    assert "database_stats" in d_metrics


def test_system_service_get_full_telemetry(tmp_path):
    test_db = str(tmp_path / "direct_telemetry.db")
    database.init_db(test_db)
    database.save_player("p2", "PlayerTwo", "KR", db_path=test_db)

    service = SystemService()
    telemetry = service.get_full_telemetry(db_path=test_db)

    assert telemetry["players"] == 1
    assert telemetry["matches"] == 0
    assert telemetry["training_records"] == 0
    assert telemetry["success"] is True
    assert telemetry["ok"] is True
    assert isinstance(telemetry["db_size_mb"], float)
    assert telemetry["crawl_queue_pending"] == 0

    regions = service.get_region_distribution(db_path=test_db)
    assert isinstance(regions, dict)
