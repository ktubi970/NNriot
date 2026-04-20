import pytest
import requests
import json
import time

BASE_URL = "http://localhost:5000"


def test_index_page():
    """Verify the main landing page loads."""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert "NNriot" in response.text


def test_api_status():
    """Check if the TF model is loaded and dimension is correct."""
    response = requests.get(f"{BASE_URL}/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["tensorflow_version"] != "N/A"
    assert data["input_dimension"] == 100000


def test_db_health():
    """Check database health metrics."""
    response = requests.get(f"{BASE_URL}/api/db-health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "matches" in data
    assert "training_records" in data
    assert "trained_records" in data
    assert "untrained_records" in data
    assert "trained_ratio" in data


def test_player_search():
    """Test searching for players (even if data is sparse)."""
    # Search for a common name fragment or just something that won't 500
    response = requests.get(f"{BASE_URL}/api/players/search?q=test")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_sample_match():
    """Verify the sample match data structure is still valid."""
    response = requests.get(f"{BASE_URL}/api/sample-match")
    assert response.status_code == 200
    data = response.json()
    assert "match_id" in data
    assert "participants" in data
    assert len(data["participants"]) >= 2


def test_predict_standard_match():
    """Test the main prediction endpoint with the sample match data."""
    # First get a valid sample match
    sample = requests.get(f"{BASE_URL}/api/sample-match").json()

    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=sample,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predicted_outcome" in data
    assert "win_probability" in data
    assert "confidence" in data


def test_predict_custom_match_fail_missing_players():
    """Verify custom prediction fails if players aren't in DB."""
    payload = {
        "blue_team": [
            {"puuid": "non_existent_1"},
            {"puuid": "2"},
            {"puuid": "3"},
            {"puuid": "4"},
            {"puuid": "5"},
        ],
        "red_team": [
            {"puuid": "6"},
            {"puuid": "7"},
            {"puuid": "8"},
            {"puuid": "9"},
            {"puuid": "10"},
        ],
        "game_mode": "CLASSIC",
    }
    response = requests.post(
        f"{BASE_URL}/api/predict/custom",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    # It should return 404 because players are missing
    assert response.status_code == 404
    assert "error" in response.json()


def test_history_status():
    """Check history collection status endpoint."""
    response = requests.get(f"{BASE_URL}/api/history/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_monitor_metrics():
    """Verify monitor metrics are populated."""
    response = requests.get(f"{BASE_URL}/api/monitor/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "database_stats" in data or "matches" in data
    assert "training_config" in data


def test_manual_train_trigger():
    """Test manual training step trigger."""
    response = requests.post(f"{BASE_URL}/api/train/manual")
    # This might return 404 if no untrained data, which is acceptable
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        assert response.json()["success"] is True
