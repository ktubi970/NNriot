"""Test the RiotAPI timeline endpoint with mocked HTTP."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("RIOT_API_KEY", "test_key_123")


def _fake_timeline_response():
    """Synthetic Match-V5 timeline payload."""
    return {
        "metadata": {"matchId": "EUW1_1234567890"},
        "info": {
            "frames": [
                {
                    "timestamp": 60000,
                    "events": [
                        {"type": "CHAMPION_KILL", "timestamp": 65000, "killerId": 1, "victimId": 6}
                    ],
                    "participantFrames": {}
                }
            ],
            "participants": [{"participantId": i, "puuid": f"p{i}"} for i in range(1, 11)]
        }
    }


def test_get_match_timeline_success():
    from riot_api import RiotAPI
    api = RiotAPI(region="EUW")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _fake_timeline_response()
    mock_resp.headers = {}

    with patch("riot_api.requests.get", return_value=mock_resp):
        result = api.get_match_timeline("EUW1_1234567890")

    assert result is not None
    assert result["metadata"]["matchId"] == "EUW1_1234567890"
    assert len(result["info"]["frames"]) == 1


def test_get_match_timeline_404_returns_none():
    from riot_api import RiotAPI
    api = RiotAPI(region="EUW")
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.headers = {}

    with patch("riot_api.requests.get", return_value=mock_resp):
        result = api.get_match_timeline("MISSING_MATCH")

    assert result is None


def test_get_match_timelines_batch():
    from riot_api import RiotAPI
    api = RiotAPI(region="EUW")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _fake_timeline_response()
    mock_resp.headers = {}

    with patch("riot_api.requests.get", return_value=mock_resp):
        results = api.get_match_timelines_batch(["M1", "M2", "M3"])

    assert len(results) == 3
    assert set(results.keys()) == {"M1", "M2", "M3"}
