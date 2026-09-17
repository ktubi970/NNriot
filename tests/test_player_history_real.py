import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from unittest.mock import patch, MagicMock
from app import app as flask_app
from core import database


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_player_history_returns_real_db_data(client, tmp_path):
    test_db = str(tmp_path / "test_history.db")
    database.init_db(test_db)
    
    # Insert a real player
    puuid = "test-puuid-123"
    database.save_player(puuid, "RealPlayer", "EUW", db_path=test_db)
    
    # Insert a real match with participant info
    match_data = {
        "metadata": {"matchId": "EUW1_9999999999"},
        "info": {
            "gameMode": "CLASSIC",
            "gameCreation": 1700000000000,
            "gameDuration": 1800,
            "participants": [
                {
                    "puuid": puuid,
                    "championName": "Ahri",
                    "teamPosition": "MIDDLE",
                    "win": True,
                    "kills": 8,
                    "deaths": 2,
                    "assists": 11,
                    "totalMinionsKilled": 210,
                    "goldEarned": 14500,
                    "item0": 3089, "item1": 3020, "item2": 0, "item3": 0, "item4": 0, "item5": 0, "item6": 3364,
                }
            ]
        }
    }
    database.save_match("EUW1_9999999999", match_data, db_path=test_db)

    with patch.dict("os.environ", {"NNRIOT_DB_PATH": test_db}):
        resp = client.get("/api/player/history?name=RealPlayer&tag=EUW&count=5")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["player"]["name"] == "RealPlayer"
        assert len(data["matches"]) == 1
        m = data["matches"][0]
        assert m["champion"] == "Ahri"
        assert m["kills"] == 8
        assert m["deaths"] == 2
        assert m["assists"] == 11
        assert m["win"] is True
        assert m["match_id"] == "EUW1_9999999999"


def test_player_history_no_matches_returns_empty_not_fake(client, tmp_path):
    test_db = str(tmp_path / "test_empty.db")
    database.init_db(test_db)

    with patch.dict("os.environ", {"NNRIOT_DB_PATH": test_db}):
        resp = client.get("/api/player/history?name=UnknownGuy&tag=NA1&count=5")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["matches"] == []
        assert data["player"]["total_games"] == 0
        # Must not fabricate Challenger LP or is_pro
        assert data["player"].get("is_pro") is False or data["player"].get("is_pro") is None


def test_player_history_sync_riot_auth_error(client, tmp_path):
    test_db = str(tmp_path / "test_auth_err.db")
    database.init_db(test_db)
    from core.riot_api import RiotAuthError

    with patch.dict("os.environ", {"NNRIOT_DB_PATH": test_db, "RIOT_API_KEY": "RGAPI-invalid"}):
        with patch("core.riot_api.RiotAPI.get_puuid", side_effect=RiotAuthError(401, "API key expired")):
            resp = client.get("/api/player/history?name=Faker&tag=KR1&sync_riot=1")
            assert resp.status_code == 503
            data = resp.get_json()
            assert data["success"] is False
            assert "error" in data
