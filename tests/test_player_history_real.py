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

    with patch.dict(flask_app.config, {"NNRIOT_DB_PATH": test_db}):
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

    with patch.dict(flask_app.config, {"NNRIOT_DB_PATH": test_db}):
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

    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-invalid"}), patch.dict(flask_app.config, {"NNRIOT_DB_PATH": test_db}):
        with patch("core.riot_api.RiotAPI.get_puuid", side_effect=RiotAuthError(401, "API key expired")):
            resp = client.get("/api/player/history?name=Faker&tag=KR1&sync_riot=1")
            assert resp.status_code == 503
            data = resp.get_json()
            assert data["success"] is False
            assert "error" in data


def test_player_history_finds_matches_beyond_recency_scan(client, tmp_path):
    """
    Verifies that a player's match history is retrieved accurately even if
    hundreds of newer matches from other players exist in the database,
    preventing the critical recency truncation bug.
    """
    test_db = str(tmp_path / "test_recency.db")
    database.init_db(test_db)
    
    target_puuid = "target-puuid-777"
    database.save_player(target_puuid, "DeepPlayer", "EUW", db_path=test_db)

    # 1. Insert target player's match at an older timestamp
    older_match = {
        "metadata": {"matchId": "EUW1_0000000001"},
        "info": {
            "gameMode": "CLASSIC",
            "gameCreation": 1600000000000,
            "gameDuration": 1500,
            "participants": [
                {
                    "puuid": target_puuid,
                    "championName": "Syndra",
                    "teamPosition": "MIDDLE",
                    "win": True,
                    "kills": 10,
                    "deaths": 1,
                    "assists": 8,
                    "totalMinionsKilled": 180,
                    "goldEarned": 12000,
                }
            ],
        },
    }
    database.save_match("EUW1_0000000001", older_match, db_path=test_db)

    # 2. Bulk insert 220 newer matches for other players
    other_matches = []
    for i in range(220):
        m = {
            "metadata": {"matchId": f"EUW1_OTHER_{i}"},
            "info": {
                "gameMode": "CLASSIC",
                "gameCreation": 1700000000000 + i * 1000,
                "gameDuration": 1200,
                "participants": [
                    {
                        "puuid": f"other-puuid-{i}",
                        "championName": "Garen",
                        "teamPosition": "TOP",
                        "win": False,
                        "kills": 0,
                        "deaths": 5,
                        "assists": 0,
                    }
                ],
            },
        }
        other_matches.append((f"EUW1_OTHER_{i}", m))
    database.save_matches_batch(other_matches, db_path=test_db)

    # 3. Query player history with count=5
    with patch.dict(flask_app.config, {"NNRIOT_DB_PATH": test_db}):
        resp = client.get("/api/player/history?name=DeepPlayer&tag=EUW&count=5")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["matches"]) == 1
        assert data["matches"][0]["champion"] == "Syndra"
        assert data["matches"][0]["kills"] == 10
        assert data["matches"][0]["match_id"] == "EUW1_0000000001"


def test_backfill_match_participants(tmp_path):
    test_db = str(tmp_path / "test_backfill.db")
    database.init_db(test_db)

    # Insert a match directly via raw SQL without match_participants
    import base64, gzip, json
    raw = {
        "info": {
            "gameMode": "CLASSIC",
            "gameCreation": 1650000000000,
            "gameDuration": 1600,
            "participants": [
                {
                    "puuid": "backfill-puuid",
                    "championName": "Zed",
                    "teamPosition": "MIDDLE",
                    "win": True,
                    "kills": 12,
                    "deaths": 3,
                    "assists": 4,
                }
            ],
        }
    }
    blob = base64.b64encode(gzip.compress(json.dumps(raw).encode("utf-8"))).decode("ascii")
    with database.get_connection(test_db) as conn:
        conn.execute(
            "INSERT INTO matches (match_id, raw_json, game_mode, game_creation_time, game_duration) VALUES (?, ?, ?, ?, ?)",
            ("EUW1_RAW_ONLY", blob, "CLASSIC", 1650000000000, 1600),
        )

    # Run backfill
    count = database.backfill_match_participants(limit=10, db_path=test_db)
    assert count == 1

    # Verify participant is now queryable
    matches = database.get_player_match_history("backfill-puuid", db_path=test_db)
    assert len(matches) == 1
    assert matches[0]["champion"] == "Zed"
    assert matches[0]["kills"] == 12

