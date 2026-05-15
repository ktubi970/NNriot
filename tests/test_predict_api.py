import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app import app as flask_app
from app import core as app_core


@pytest.fixture
def client(monkeypatch):
    """Build a Flask test client with a mocked global_trainer."""
    fake_trainer = MagicMock()
    fake_trainer.predict.return_value = _fake_preds()

    # init_trainer runs at app package import time; overwrite the
    # core-module globals via monkeypatch so the assignments are undone
    # after each test (clean teardown).
    monkeypatch.setattr(app_core, "global_trainer", fake_trainer)
    monkeypatch.setattr(app_core, "tf_available", True)
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _fake_preds():
    """Synthetic Keras-dict output for a single match (batch size 1)."""
    return {
        "winner":           np.array([[0.40, 0.60]], dtype=np.float32),
        "team_b_kill_lead":     np.array([[0.45, 0.55]], dtype=np.float32),
        "first_blood":      np.array([[0.50, 0.45, 0.05]], dtype=np.float32),
        "first_baron":      np.array([[0.40, 0.45, 0.15]], dtype=np.float32),
        "first_inhibitor":  np.array([[0.42, 0.43, 0.15]], dtype=np.float32),
        "first_tower":      np.array([[0.50, 0.48, 0.02]], dtype=np.float32),
        "kills_odd":        np.array([[0.51]], dtype=np.float32),
        "both_baron":       np.array([[0.22]], dtype=np.float32),
        "both_inhibitor":   np.array([[0.41]], dtype=np.float32),
        "both_dragon":      np.array([[0.93]], dtype=np.float32),
        "elder_dragon":     np.array([[0.18]], dtype=np.float32),
        "kill_handicap":    np.array([[-4.2]], dtype=np.float32),
        "total_kills":      np.array([[36.4]], dtype=np.float32),
        "team_a_kills":     np.array([[16.1]], dtype=np.float32),
        "team_b_kills":     np.array([[20.3]], dtype=np.float32),
        "total_barons":     np.array([[1.8]], dtype=np.float32),
        "total_dragons":    np.array([[3.6]], dtype=np.float32),
        "total_towers":     np.array([[14.2]], dtype=np.float32),
        # Timeline kill-threshold heads (3-class softmax; class 2 = "neither")
        "first_to_5_kills":  np.array([[0.30, 0.20, 0.50]], dtype=np.float32),
        "first_to_10_kills": np.array([[0.25, 0.25, 0.50]], dtype=np.float32),
        "first_to_15_kills": np.array([[0.10, 0.10, 0.80]], dtype=np.float32),
        "first_to_20_kills": np.array([[0.05, 0.05, 0.90]], dtype=np.float32),
    }


def _sample_match_payload():
    """Minimal payload matching the validation in /api/predict."""
    return {
        "participants": [
            {"teamId": 100, "championName": "Aatrox", "kills": 5, "deaths": 3, "assists": 7, "goldEarned": 12000, "teamPosition": "TOP"},
            {"teamId": 200, "championName": "Lux",    "kills": 4, "deaths": 6, "assists": 8, "goldEarned": 11000, "teamPosition": "MID"},
        ],
        "teams": [
            {"teamId": 100, "win": False},
            {"teamId": 200, "win": True},
        ],
    }


def test_predict_returns_multi_output_shape(client):
    resp = client.post("/api/predict",
                       data=json.dumps(_sample_match_payload()),
                       content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    # New shape
    assert "winner" in data and "team_a" in data["winner"] and "team_b" in data["winner"]
    assert data["winner"]["predicted"] in ("A", "B")
    assert "first" in data and set(data["first"].keys()) == {
        "blood", "baron", "inhibitor", "tower",
        "kills_5", "kills_10", "kills_15", "kills_20",
    }
    assert "kills" in data and set(data["kills"].keys()) == {"total", "team_a", "team_b", "handicap", "odd_probability"}
    assert "totals" in data and set(data["totals"].keys()) == {"barons", "dragons", "towers"}
    assert "both_teams" in data and set(data["both_teams"].keys()) == {"baron", "inhibitor", "dragon"}
    assert "elder_dragon" in data
    # Legacy shim has been removed — these keys must NOT be present
    assert "predicted_outcome" not in data
    assert "win_probability" not in data
    assert "lose_probability" not in data


def test_predict_winner_predicted_field(client):
    """winner.predicted is 'B' when team_b prob > team_a prob."""
    resp = client.post("/api/predict",
                       data=json.dumps(_sample_match_payload()),
                       content_type="application/json")
    data = resp.get_json()
    # Fake preds: winner team_a=0.4, team_b=0.6 → predicted B
    assert data["winner"]["predicted"] == "B"
    assert data["winner"]["team_a"] == 0.4
    assert data["winner"]["team_b"] == 0.6


def test_predict_first_event_three_keys(client):
    resp = client.post("/api/predict",
                       data=json.dumps(_sample_match_payload()),
                       content_type="application/json")
    data = resp.get_json()
    # 4 first-event keys + 4 timeline kill-threshold keys = 8 sub-keys total.
    for event in ("blood", "baron", "inhibitor", "tower",
                  "kills_5", "kills_10", "kills_15", "kills_20"):
        assert set(data["first"][event].keys()) == {"team_a", "team_b", "none"}


def test_predict_handles_missing_fields(client):
    """Missing 'participants' or 'teams' must return 400."""
    payload = {"participants": [{"teamId": 100}]}  # no 'teams'
    resp = client.post("/api/predict",
                       data=json.dumps(payload),
                       content_type="application/json")
    assert resp.status_code == 400


def test_predict_handles_model_unavailable(client, monkeypatch):
    """When tf_available=False, /api/predict returns 500 with error."""
    monkeypatch.setattr(app_core, "tf_available", False)
    monkeypatch.setattr(app_core, "global_trainer", None)
    resp = client.post("/api/predict",
                       data=json.dumps(_sample_match_payload()),
                       content_type="application/json")
    assert resp.status_code == 500
