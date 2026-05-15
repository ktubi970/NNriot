"""Flask test client tests for the main API surface.

Replaces the previous live-server tests at localhost:5000. Uses
monkeypatched MagicMock trainer to avoid loading TF at test time
(other than via the initial app package import).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from app import app as flask_app
from app import core as app_core


def _fake_preds():
    """Synthetic Keras dict output for the 22-head multi-output model."""
    return {
        "winner":            np.array([[0.40, 0.60]], dtype=np.float32),
        "team_b_kill_lead":  np.array([[0.45, 0.55]], dtype=np.float32),
        "first_blood":       np.array([[0.50, 0.45, 0.05]], dtype=np.float32),
        "first_baron":       np.array([[0.40, 0.45, 0.15]], dtype=np.float32),
        "first_inhibitor":   np.array([[0.42, 0.43, 0.15]], dtype=np.float32),
        "first_tower":       np.array([[0.50, 0.48, 0.02]], dtype=np.float32),
        "kills_odd":         np.array([[0.51]], dtype=np.float32),
        "both_baron":        np.array([[0.22]], dtype=np.float32),
        "both_inhibitor":    np.array([[0.41]], dtype=np.float32),
        "both_dragon":       np.array([[0.93]], dtype=np.float32),
        "elder_dragon":      np.array([[0.18]], dtype=np.float32),
        "kill_handicap":     np.array([[0.0]], dtype=np.float32),  # normalized scale
        "total_kills":       np.array([[0.3]], dtype=np.float32),
        "team_a_kills":      np.array([[0.1]], dtype=np.float32),
        "team_b_kills":      np.array([[0.5]], dtype=np.float32),
        "total_barons":      np.array([[0.2]], dtype=np.float32),
        "total_dragons":     np.array([[0.4]], dtype=np.float32),
        "total_towers":      np.array([[0.3]], dtype=np.float32),
        "first_to_5_kills":  np.array([[0.45, 0.40, 0.15]], dtype=np.float32),
        "first_to_10_kills": np.array([[0.40, 0.40, 0.20]], dtype=np.float32),
        "first_to_15_kills": np.array([[0.20, 0.20, 0.60]], dtype=np.float32),
        "first_to_20_kills": np.array([[0.10, 0.10, 0.80]], dtype=np.float32),
    }


@pytest.fixture
def client(monkeypatch):
    """Flask test client with the trainer mocked out (no TF predict cost)."""
    fake_trainer = MagicMock()
    fake_trainer.predict.return_value = _fake_preds()
    # JSON-serializable return for /api/train/manual.
    fake_trainer.run_training_step.return_value = 0

    monkeypatch.setattr(app_core, "global_trainer", fake_trainer)
    monkeypatch.setattr(app_core, "tf_available", True)
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _sample_match():
    """Minimal /api/predict payload."""
    return {
        "participants": [
            {"teamId": 100, "championName": "Aatrox", "kills": 10, "deaths": 5,
             "assists": 7, "goldEarned": 14000, "teamPosition": "TOP"},
            {"teamId": 200, "championName": "Lux", "kills": 8, "deaths": 7,
             "assists": 4, "goldEarned": 12000, "teamPosition": "MID"},
        ],
        "teams": [
            {"teamId": 100, "win": True},
            {"teamId": 200, "win": False},
        ],
    }


# ============================================================================
# Static pages — confirm they render
# ============================================================================

def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"NNriot" in resp.data


def test_explorer_page(client):
    assert client.get("/explorer").status_code == 200


def test_predictor_page(client):
    assert client.get("/predictor").status_code == 200


def test_monitor_page(client):
    assert client.get("/monitor").status_code == 200


# ============================================================================
# Health / status — current shape (NO legacy fields)
# ============================================================================

def test_api_status(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["model_loaded"] is True
    assert "tensorflow_version" in data
    # New VECTOR_DIM (was 100000 before Sprint 2)
    assert data["input_dimension"] == 20000


def test_db_health(client):
    resp = client.get("/api/db-health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "matches" in data
    assert "training_records" in data


def test_monitor_metrics(client):
    resp = client.get("/api/monitor/metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "matches" in data
    assert "training_config" in data


# ============================================================================
# Players
# ============================================================================

def test_player_search_short_query(client):
    """Queries < 2 chars return []."""
    resp = client.get("/api/players/search?q=a")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_player_search_valid(client):
    resp = client.get("/api/players/search?q=test")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


# ============================================================================
# Predict — modern multi-output shape (NO legacy shim)
# ============================================================================

def test_sample_match(client):
    resp = client.get("/api/sample-match")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "participants" in data
    assert len(data["participants"]) >= 2


def test_predict_returns_multi_output(client):
    resp = client.post("/api/predict",
                       data=json.dumps(_sample_match()),
                       content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    # New shape
    assert "winner" in data
    assert "team_b_kill_lead" in data
    assert "kills" in data
    assert "first" in data
    assert "totals" in data
    assert "both_teams" in data
    assert "elder_dragon" in data
    # Legacy shim removed in 7840e87 — must NOT be present
    assert "predicted_outcome" not in data
    assert "win_probability" not in data
    assert "lose_probability" not in data


def test_predict_missing_fields(client):
    resp = client.post("/api/predict",
                       data=json.dumps({"participants": []}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_predict_custom_invalid_team_size(client):
    """5v5 required — fewer players returns 400."""
    payload = {
        "blue_team": [{"puuid": "p1"}],
        "red_team":  [{"puuid": "p2"}],
        "game_mode": "CLASSIC",
    }
    resp = client.post("/api/predict/custom",
                       data=json.dumps(payload),
                       content_type="application/json")
    assert resp.status_code == 400


# ============================================================================
# Async / status endpoints
# ============================================================================

def test_history_status(client):
    resp = client.get("/api/history/status")
    assert resp.status_code == 200
    assert "status" in resp.get_json()


def test_train_manual(client):
    """Trigger returns 200 with success, or 404 if no untrained data."""
    resp = client.post("/api/train/manual")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        assert resp.get_json()["success"] is True


# ============================================================================
# Stream OCR gating
# ============================================================================

def test_stream_ocr_disabled_by_default(client):
    """Without MOCK_OCR_ENABLED=1 env var, /api/stream/process_frame returns 503."""
    resp = client.post("/api/stream/process_frame",
                       data=json.dumps({"image": "data:image/jpeg;base64,/9j/"}),
                       content_type="application/json")
    assert resp.status_code == 503
