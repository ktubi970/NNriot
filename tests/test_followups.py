import os
import sys

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_collector
import database


def _match(team_a_win, team_b_win):
    return {
        "info": {
            "gameVersion": "14.10.400",
            "gameMode": "CLASSIC",
            "participants": [
                {"championName": "Aatrox", "teamId": 100, "kills": 1, "deaths": 1, "assists": 1, "goldEarned": 1000, "teamPosition": "TOP"},
                {"championName": "Lux", "teamId": 200, "kills": 1, "deaths": 1, "assists": 1, "goldEarned": 1000, "teamPosition": "MID"},
            ],
            "teams": [
                {"teamId": 100, "win": team_a_win},
                {"teamId": 200, "win": team_b_win},
            ],
        }
    }


def test_build_training_record_team_a_wins():
    rec = data_collector._build_training_record("TEST_1", _match(True, False))
    assert rec is not None
    assert rec[0] == "TEST_1"
    assert rec[2] == 0  # team A label


def test_build_training_record_team_b_wins():
    rec = data_collector._build_training_record("TEST_2", _match(False, True))
    assert rec is not None
    assert rec[2] == 1


def test_build_training_record_skips_no_winner():
    rec = data_collector._build_training_record("TEST_3", _match(False, False))
    assert rec is None


def test_get_untrained_records_returns_dict(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)
    # Insert a match (FK requirement) and a training record
    with database.get_connection(db) as conn:
        conn.execute("INSERT INTO matches (match_id, game_mode) VALUES (?, ?)", ("M1", "CLASSIC"))
    database.save_training_record("M1", {"foo": "bar", "n": 42}, 0, db_path=db)

    records = database.get_untrained_records(limit=10, db_path=db)
    assert len(records) == 1
    assert isinstance(records[0]["feature_json"], dict)
    assert records[0]["feature_json"]["foo"] == "bar"
    assert records[0]["feature_json"]["n"] == 42
