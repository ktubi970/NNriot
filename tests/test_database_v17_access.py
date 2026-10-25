"""Schema v17 access: derived player statistics, trainer state, sightings and times."""

import sqlite3
import time

import pytest

from core import database


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "v17.db")
    database.init_db(path)
    return path


def _participants(db, rows):
    with sqlite3.connect(db) as conn:
        conn.executemany("INSERT OR IGNORE INTO matches (match_id) VALUES (?)", {(r[0],) for r in rows})
        conn.executemany("INSERT INTO match_participants (match_id, puuid, kills, deaths, assists, gold_earned) "
                         "VALUES (?, ?, ?, ?, ?, ?)", rows)


def test_player_statistics_are_derived_from_participants(db):
    database.save_player("main", "Main", "EUW", db_path=db)
    database.save_player("smurf", "Smurf", "EUW", db_path=db)
    database.save_player("idle", "Idle", "EUW", db_path=db)
    database.link_accounts("main", "smurf", db_path=db)
    _participants(db, [("M1", "main", 8, 0, 3, 12000), ("M2", "main", None, None, 4, 10000),
                       ("M3", "smurf", 2, 2, 1, None), ("M4", "other", 50, 0, 0, 99000)])

    stats = database.get_player_stats("smurf", db_path=db)
    assert stats == {"puuid": "main", "game_name": "Main", "tag_line": "EUW",
                     "total_matches": 3, "avg_kda": 9.0, "avg_gold": 7333}
    assert database.get_player_by_name_tag("main", "euw", db_path=db)["total_matches"] == 2
    assert database.get_player_stats("idle", db_path=db)["avg_kda"] == 0.0
    assert database.get_player_stats("absent", db_path=db) is None
    assert [(r["puuid"], r["total_matches"]) for r in database.get_top_players(2, db_path=db)] == [
        ("main", 2), ("smurf", 1)]
    found = database.search_players("Ma", db_path=db)
    assert [(r["puuid"], r["total_matches"], r["avg_gold"]) for r in found] == [("main", 2, 11000)]


def test_player_profile_refresh_is_recorded_in_milliseconds(db):
    before = int(time.time() * 1000)
    database.save_player("fresh", "Fresh", "EUW", db_path=db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO players (puuid, updated_at) VALUES ('old', ?)", (before - 8 * 86_400_000,))
        conn.execute("INSERT INTO players (puuid) VALUES ('never')")
        updated = conn.execute("SELECT updated_at FROM players WHERE puuid = 'fresh'").fetchone()[0]
    assert before <= updated <= int(time.time() * 1000)
    assert sorted(database.get_stale_puuids(["fresh", "old", "never", "absent"], db_path=db)) == [
        "absent", "never", "old"]


def test_trainer_state_lives_outside_training_rows(db):
    with sqlite3.connect(db) as conn:
        conn.executemany("INSERT INTO matches (match_id) VALUES (?)", [("M1",), ("M2",), ("M3",)])
    database.save_training_records_batch(
        [("M1", {"team_a": []}, 1), ("M2", {"team_a": []}, 0), ("M3", {"team_a": []}, 1)],
        db_path=db, feature_extractor="json_utils.extract_pre_match_features")
    ids = {r["match_id"]: r["id"] for r in database.get_untrained_records(10, db_path=db)}
    before = int(time.time() * 1000)

    database.mark_as_trained([ids["M1"], ids["M3"], 999], db_path=db)

    assert [r["match_id"] for r in database.get_untrained_records(10, db_path=db)] == ["M2"]
    assert sorted(r["match_id"] for r in database.get_random_trained_records(10, db_path=db)) == ["M1", "M3"]
    stats = database.get_db_stats(db_path=db)
    assert (stats["trained_records"], stats["untrained_records"], stats["schema_version"]) == (2, 1, 17)
    with sqlite3.connect(db) as conn:
        trained = dict(conn.execute("SELECT training_id, trained_at FROM trained_records"))
        extractors = {r[0] for r in conn.execute("SELECT feature_extractor FROM training_dataset")}
    assert set(trained) == {ids["M1"], ids["M3"]} and min(trained.values()) >= before
    assert extractors == {"json_utils.extract_pre_match_features"}


def test_sightings_record_every_fetch_and_remember_discovery_only_matches(db):
    database.save_match_sightings([("EUW1_1", "seed", "discovery_only"), ("EUW1_2", "seed", "stored"),
                                   ("EUW1_1", "p1", "discovery_only")], db_path=db, seen_at=5000)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT match_id, fetched_by_puuid, seen_at, disposition FROM match_sightings").fetchall()
    assert len(rows) == 3 and {r[2] for r in rows} == {5000}
    assert database.discovery_only_match_ids(["EUW1_1", "EUW1_2", "EUW1_3"], db_path=db) == {"EUW1_1"}
    with pytest.raises(sqlite3.IntegrityError):
        database.save_match_sightings([("EUW1_3", None, "kept")], db_path=db)


def test_legacy_crawl_queue_helpers_write_milliseconds(db):
    database.add_to_crawl_queue("Legacy", "EUW", db_path=db)
    database.mark_as_processed("Legacy", "EUW", db_path=db)
    with sqlite3.connect(db) as conn:
        created, processed = conn.execute("SELECT created_at, processed_at FROM crawl_queue").fetchone()
    assert type(created) is int and type(processed) is int and processed >= created
