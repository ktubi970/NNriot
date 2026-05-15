"""Test the v10 -> v11 migration: match_timelines table."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database


def test_v11_migration_creates_match_timelines_table(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)

    with database.get_connection(db) as conn:
        cols = conn.execute("PRAGMA table_info(match_timelines)").fetchall()

    col_names = {c["name"] for c in cols}
    assert col_names == {"match_id", "timeline_json"}


def test_save_and_get_match_timeline(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)

    # Insert a parent match (FK)
    with database.get_connection(db) as conn:
        conn.execute(
            "INSERT INTO matches (match_id, game_mode) VALUES (?, ?)",
            ("M1", "CLASSIC")
        )

    timeline_data = {"metadata": {"matchId": "M1"}, "info": {"frames": []}}
    database.save_match_timeline("M1", timeline_data, db_path=db)

    fetched = database.get_match_timeline("M1", db_path=db)
    assert fetched == timeline_data
    assert database.match_timeline_exists("M1", db_path=db) is True
    assert database.match_timeline_exists("M99", db_path=db) is False


def test_save_match_timelines_batch(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)

    with database.get_connection(db) as conn:
        for i in range(3):
            conn.execute(
                "INSERT INTO matches (match_id, game_mode) VALUES (?, ?)",
                (f"M{i}", "CLASSIC")
            )

    timelines = [(f"M{i}", {"info": {"frames": [{"timestamp": i * 1000}]}}) for i in range(3)]
    database.save_match_timelines_batch(timelines, db_path=db)

    for i in range(3):
        t = database.get_match_timeline(f"M{i}", db_path=db)
        assert t["info"]["frames"][0]["timestamp"] == i * 1000


def test_schema_version_now_11():
    assert database.SCHEMA_VERSION == 11
