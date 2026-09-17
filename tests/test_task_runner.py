import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from unittest.mock import MagicMock
import pytest
from core.task_runner import TaskRunner, global_task_runner
from app import app as flask_app
from app import core as app_core


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
    time.sleep(0.1)
    status = runner.get_status(job_id)
    assert status["status"] == "completed"
    assert status["result"] == 30
    assert status["error"] is None
    runner.shutdown()


def test_task_runner_records_failure():
    runner = TaskRunner(max_workers=2)

    def failing_job():
        raise ValueError("Boom")

    job_id = runner.submit_task("fail_job", failing_job)
    time.sleep(0.1)
    status = runner.get_status(job_id)
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


def test_api_jobs_endpoint():
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


def test_api_train_returns_job_id(monkeypatch):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    monkeypatch.setattr(app_core, "tf_available", False)
    payload = {"matches": [{"matchId": "EUW1_12345"}]}
    resp = client.post("/api/train", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "job_id" in data
    assert data["job_id"].startswith("job_")

    job_status = global_task_runner.get_status(data["job_id"])
    assert job_status is not None
    assert job_status["type"] == "training"


def test_history_and_batch_endpoints_with_task_runner(monkeypatch):
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
