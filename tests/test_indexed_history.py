"""History reads never repair a database as a side effect."""
import json
import sqlite3

import pytest

from core import database


def _raw(created=1700000000000):
    return {"info": {"gameMode": "CLASSIC", "gameCreation": created,
        "gameDuration": 1800, "participants": [{"puuid": "player-1",
        "championName": "Ahri", "teamPosition": "MIDDLE", "win": True,
        "kills": 8, "deaths": 2, "assists": 4, "totalMinionsKilled": 200,
        "goldEarned": 12000}]}}


def test_raw_only_history_requires_explicit_backfill(tmp_path):
    path = str(tmp_path / "history.db")
    database.init_db(path)
    with database.get_connection(path) as conn:
        conn.execute("INSERT INTO matches (match_id, raw_json, game_mode) VALUES (?,?,?)",
                     ("EUW1_RAW", json.dumps(_raw()), "CLASSIC"))
    assert database.get_player_match_history("player-1", db_path=path) == []
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM match_participants").fetchone()[0] == 0
    assert database.backfill_match_participants(db_path=path) == 1
    result = database.get_player_match_history("player-1", db_path=path)
    assert len(result) == 1
    assert result[0]["match_id"] == "EUW1_RAW"
    assert result[0]["kda"] == "6.0"


def test_indexed_history_preserves_order_limit_and_display(tmp_path):
    path = str(tmp_path / "history.db")
    database.init_db(path)
    database.save_match("EUW1_OLD", _raw(), db_path=path)
    database.save_match("EUW1_NEW", _raw(1700000001000), db_path=path)
    rows = database.get_player_match_history("player-1", limit=1, db_path=path)
    assert [row["match_id"] for row in rows] == ["EUW1_NEW"]
    assert rows[0]["duration"] == "30:00"
    assert rows[0]["cs_per_min"] == "6.7"
    assert database.get_player_match_history("absent", db_path=path) == []


def test_missing_participant_table_is_a_setup_error(tmp_path):
    path = str(tmp_path / "unmigrated.db")
    database.init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE match_participants")
    with pytest.raises(sqlite3.OperationalError, match="match_participants"):
        database.get_player_match_history("player-1", db_path=path)
