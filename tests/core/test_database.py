"""Test the v10 -> v11 migration: match_timelines table."""

import pytest


from core import database
from core.repository import MatchRepository, SystemRepository, TrainingRepository


@pytest.mark.integration
def test_v11_migration_creates_match_timelines_table(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)

    with database.get_connection(db) as conn:
        cols = conn.execute("PRAGMA table_info(match_timelines)").fetchall()

    col_names = {c["name"] for c in cols}
    assert col_names == {"match_id", "timeline_json"}


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.unit
def test_schema_version_is_at_least_12():
    assert database.SCHEMA_VERSION >= 12
    assert database.SCHEMA_VERSION == 12


@pytest.mark.integration
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


@pytest.mark.integration
def test_repository_saves_training_record_and_reports_required_stats(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)

    match_repo = MatchRepository(db)
    training_repo = TrainingRepository(db)
    system_repo = SystemRepository(db)

    with match_repo.get_connection() as conn:
        conn.execute(
            "INSERT INTO matches (match_id, game_mode) VALUES (?, ?)",
            ("TEST_MATCH", "CLASSIC"),
        )

    training_repo.save_training_records_batch(
        [("TEST_MATCH", {"foo": "bar"}, 0, {"test": True})]
    )

    stats = system_repo.get_db_stats()
    required_keys = {
        "players",
        "matches",
        "trained_records",
        "untrained_records",
        "trained_ratio",
    }
    assert required_keys <= stats.keys()

    with training_repo.get_connection() as conn:
        row = conn.execute(
            "SELECT match_id, feature_json FROM training_dataset WHERE match_id = ?",
            ("TEST_MATCH",),
        ).fetchone()

    assert row is not None
    assert row["match_id"] == "TEST_MATCH"
    assert row["feature_json"]
