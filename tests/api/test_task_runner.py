import time
from unittest.mock import MagicMock
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from core.task_runner import TaskRunner, global_task_runner


@pytest.fixture
def app_modules():
    from app import app as flask_app
    from app import core as app_core

    return flask_app, app_core


def test_task_runner_executes_and_tracks_job():
    runner = TaskRunner(max_workers=2)

    def simple_job(x, y):
        time.sleep(0.05)
        return x + y

    job_id = runner.submit_task("test_add", simple_job, 10, 20)
    assert job_id is not None
    status = runner.get_status(job_id)
    assert status["status"] in ("queued", "running", "completed")

    # Wait for completion
    for _ in range(50):
        status = runner.get_status(job_id)
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "completed"
    assert status["result"] == 30
    assert status["error"] is None
    runner.shutdown()


def test_task_runner_records_failure():
    runner = TaskRunner(max_workers=2)

    def failing_job():
        raise ValueError("Boom")

    job_id = runner.submit_task("fail_job", failing_job)
    for _ in range(50):
        status = runner.get_status(job_id)
        if status["status"] == "failed":
            break
        time.sleep(0.05)
    assert status["status"] == "failed"
    assert "Boom" in status["error"]
    runner.shutdown()


def test_task_runner_get_latest_by_type():
    runner = TaskRunner(max_workers=2)
    assert runner.get_latest_by_type("nonexistent") is None

    j1 = runner.submit_task("task_a", lambda: 1)
    time.sleep(0.01)
    j2 = runner.submit_task("task_a", lambda: 2)
    latest = runner.get_latest_by_type("task_a")
    assert latest is not None
    assert latest["job_id"] == j2
    runner.shutdown()


def test_task_runner_memory_bounding_prunes_oldest_jobs():
    runner = TaskRunner(max_workers=2, max_history=3)
    j1 = runner.submit_task("t", lambda: 1)
    j2 = runner.submit_task("t", lambda: 2)
    j3 = runner.submit_task("t", lambda: 3)
    time.sleep(0.05)
    assert len(runner.jobs) == 3

    # 4th job triggers pruning of oldest completed
    j4 = runner.submit_task("t", lambda: 4)
    time.sleep(0.05)
    assert len(runner.jobs) <= 3
    assert runner.get_status(j1) is None
    assert runner.get_status(j4) is not None
    runner.shutdown()


def test_api_jobs_endpoint(app_modules):
    flask_app, app_core = app_modules

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    # 404 for unknown job
    resp = client.get("/api/jobs/job_unknown_123")
    assert resp.status_code == 404
    assert resp.get_json()["success"] is False

    # 200 for submitted job
    job_id = global_task_runner.submit_task("dummy_task", lambda: 42)
    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["job"]["job_id"] == job_id
    assert data["job"]["type"] == "dummy_task"


def test_api_train_returns_job_id(monkeypatch, app_modules):
    flask_app, app_core = app_modules

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    monkeypatch.setattr(app_core, "tf_available", False)
    payload = {"matches": [{"matchId": "EUW1_12345"}]}
    
    # 1. Veracity: Without ?simulate=true, fail-fast with 503
    resp_fail = client.post("/api/train", json=payload)
    assert resp_fail.status_code == 503
    data_fail = resp_fail.get_json()
    assert data_fail["success"] is False
    assert "unavailable" in data_fail["error"]

    # 2. With ?simulate=true, explicit mock simulation allowed
    resp = client.post("/api/train?simulate=true", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "job_id" in data
    assert data["job_id"].startswith("job_")

    job_status = global_task_runner.get_status(data["job_id"])
    assert job_status is not None
    assert job_status["type"] == "training"


def test_task_runner_handles_non_serializable_result():
    runner = TaskRunner(max_workers=1)
    
    class UnserializableObject:
        def __str__(self):
            return "<UnserializableObjectInstance>"

    job_id = runner.submit_task("custom", lambda: UnserializableObject())
    for _ in range(50):
        status = runner.get_status(job_id)
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "completed"
    assert status["result"] == "<UnserializableObjectInstance>"
    runner.shutdown(wait=True)


def test_history_and_batch_endpoints_with_task_runner(monkeypatch, app_modules):
    flask_app, app_core = app_modules

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    from scripts import data_collector
    monkeypatch.setattr(data_collector, "collect_training_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_collector, "collect_batch_with_smurfs", lambda *args, **kwargs: None)

    # Test /api/history/collect
    with app_core.collection_lock:
        app_core.collection_status["status"] = "idle"
    resp = client.post("/api/history/collect", json={"game_name": "TestPlayer", "tag_line": "EUW", "count": 1})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "job_id" in data

    # Test /api/history/status
    resp = client.get("/api/history/status")
    assert resp.status_code == 200
    st = resp.get_json()
    assert "status" in st

    # Test /api/batch/collect
    with data_collector.batch_lock:
        data_collector.batch_status["status"] = "idle"
    resp = client.post("/api/batch/collect", json={"players": "Player#EUW", "sources": ["lolpros"], "count": 1})
    assert resp.status_code == 200
    b_data = resp.get_json()
    assert b_data["success"] is True
    assert "job_id" in b_data

    # Test /api/batch/status
    resp = client.get("/api/batch/status")
    assert resp.status_code == 200
    b_st = resp.get_json()
    assert "status" in b_st


def test_route_failure_propagates_to_task_runner(monkeypatch, app_modules):
    flask_app, app_core = app_modules

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    from scripts import data_collector
    def failing_collector(*args, **kwargs):
        raise ValueError("Simulated route failure in collector")

    monkeypatch.setattr(data_collector, "collect_training_data", failing_collector)

    with app_core.collection_lock:
        app_core.collection_status["status"] = "idle"

    resp = client.post("/api/history/collect", json={"game_name": "CrashPlayer", "tag_line": "EUW", "count": 1})
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]

    # Wait for the task to execute and fail
    time.sleep(0.1)
    status = global_task_runner.get_status(job_id)
    assert status is not None
    assert status["status"] == "failed"
    assert "Simulated route failure in collector" in (status["error"] or "")

    # Check /api/jobs/<job_id> reflects the failure
    job_resp = client.get(f"/api/jobs/{job_id}")
    assert job_resp.status_code == 200
    assert job_resp.get_json()["job"]["status"] == "failed"


def test_history_status_picks_latest_by_created_at(app_modules):
    flask_app, app_core = app_modules

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    j_old = global_task_runner.submit_task("history_collect", lambda: None)
    time.sleep(0.02)
    j_new = global_task_runner.submit_task("history_collect_batch", lambda: None)
    time.sleep(0.02)

    resp = client.get("/api/history/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("job_id") == j_new


def test_lolpros_import_uses_task_runner(monkeypatch, app_modules):
    flask_app, app_core = app_modules

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    from scripts import data_collector
    from core import database
    monkeypatch.setattr(database, "save_player", lambda *args, **kwargs: None)
    monkeypatch.setattr(database, "link_accounts", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_collector, "collect_by_puuid", lambda *args, **kwargs: {"matches_saved": 0})

    with app_core.collection_lock:
        app_core.collection_status["status"] = "idle"

    payload = {
        "player": {
            "name": "Faker",
            "league_player": {
                "accounts": [
                    {
                        "encrypted_puuid": "fake_puuid_123",
                        "server": "KR",
                        "gamename": "Hide on bush",
                        "tagline": "KR1",
                        "rank": {"score": 100},
                    }
                ]
            }
        }
    }

    resp = client.post("/api/lolpros/import", json=payload)
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True

    latest = global_task_runner.get_latest_by_type("lolpros_import")
    assert latest is not None
