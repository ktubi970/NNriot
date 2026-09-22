import pytest

pytestmark = pytest.mark.integration


import sqlite3
from core import database

def test_purge_polluted_training_data(tmp_path):
    db_path = str(tmp_path / "test_purge.db")
    database.init_db(db_path)

    with database.get_connection(db_path) as conn:
        # 1 valid match
        conn.execute("INSERT INTO matches (match_id, game_mode) VALUES ('VALID_1', 'CLASSIC')")
        conn.execute("INSERT INTO training_dataset (match_id, feature_json, winner_label) VALUES ('VALID_1', '{}', 1)")
        # 1 ARAM match
        conn.execute("INSERT INTO matches (match_id, game_mode) VALUES ('ARAM_1', 'ARAM')")
        conn.execute("INSERT INTO training_dataset (match_id, feature_json, winner_label) VALUES ('ARAM_1', '{}', 0)")
        # 1 TEST_MATCH
        conn.execute("INSERT INTO matches (match_id, game_mode) VALUES ('TEST_MATCH', 'CLASSIC')")
        conn.execute("INSERT INTO training_dataset (match_id, feature_json, winner_label) VALUES ('TEST_MATCH', '{}', 1)")

    summary = database.purge_polluted_training_data(db_path=db_path)

    assert summary["deleted_test_matches"] == 1
    assert summary["deleted_polluted_training_records"] == 2  # ARAM + TEST_MATCH
    
    with database.get_connection(db_path) as conn:
        rem_td = conn.execute("SELECT match_id FROM training_dataset").fetchall()
        assert len(rem_td) == 1
        assert rem_td[0]["match_id"] == "VALID_1"
