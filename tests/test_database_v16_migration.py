"""The v15 -> v16 migration: provenance tables, match facts and integer-millisecond times."""

import json
import sqlite3
import time

import pytest

from core import database

SEED_ENTRY = "2026-10-11 09:25:00"
SEED_ENTRY_MS = 1791710700000


def _payload(match_id="EUW1_1", creation=1_700_000_000_000):
    participants = [{"puuid": f"puuid_{i}", "teamId": 100 if i < 5 else 200,
                     "championName": "Ahri", "win": i < 5} for i in range(10)]
    return {"metadata": {"matchId": match_id}, "info": {
        "gameMode": "CLASSIC", "queueId": 420, "platformId": "EUW1",
        "gameCreation": creation, "gameStartTimestamp": creation + 30_000,
        "gameEndTimestamp": creation + 1_830_000, "gameDuration": 1800,
        "gameVersion": "16.18.1", "participants": participants}}


def _v15_database(path):
    with sqlite3.connect(path) as conn:
        for from_version, statements in database._MIGRATIONS:
            if from_version >= 15:
                break
            for sql in statements:
                conn.execute(sql)
    return str(path)


def _legacy_rows(db):
    # A plain connection does not enforce foreign keys, like the scripts that wrote legacy rows.
    with sqlite3.connect(db) as conn:
        conn.executemany("INSERT INTO matches (match_id, game_mode) VALUES (?, 'CLASSIC')",
                         [("EUW1_1",), ("EUW1_2",), ("EUW1_3",)])
        conn.executemany(
            "INSERT INTO crawl_queue (puuid, region, priority, is_processed, created_at, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [("seed", "EUW", 10, 1, SEED_ENTRY, "2026-10-12 00:00:00"),
             ("snowball", "KR", 0, 0, "2026-10-12 00:00:01", None),
             ("deleted", "EUW", 0, 0, "2026-10-12 00:00:02", None)])
        conn.execute("DELETE FROM crawl_queue WHERE puuid = 'deleted'")
        conn.executemany(
            "INSERT INTO players (puuid, game_name, tag_line, last_updated, avg_kda, avg_gold, total_matches) "
            "VALUES (?, ?, ?, ?, 3.5, 11000, 40)",
            [("seed", "Seed", "EUW", "2026-10-11 10:00:00"), ("pro", "Pro", "KR1", "2026-05-02 08:00:00"),
             ("smurf", "Smurf", "KR1", None)])
        conn.execute("INSERT INTO player_aliases VALUES ('smurf', 'pro')")
        conn.executemany(
            "INSERT INTO training_dataset (id, match_id, feature_json, winner_label, is_trained, last_trained_at, "
            "llm_summary, labels_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(1, "EUW1_1", "H4sIAAAA", 1, 1, "2026-10-20 12:00:00", "old summary", None),
             (2, "EUW1_2", json.dumps({"team_a": [{"champion": "Ahri", "kda": 4.0, "gold": 12000}]}), 0, 0, None, None, None),
             (3, "EUW1_ORPHAN", "{}", 1, 0, None, None, None),
             (9, "EUW1_3", "{}", 0, 0, None, None, None)])
        conn.execute("DELETE FROM training_dataset WHERE id = 9")


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _rows(conn, sql):
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql)]


def test_migration_reshapes_tables_and_converts_legacy_times(tmp_path):
    db = _v15_database(tmp_path / "legacy.db")
    _legacy_rows(db)

    database.migrate_db(db)

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 16 == database.SCHEMA_VERSION
        assert _columns(conn, "players") == ["puuid", "game_name", "tag_line", "updated_at"]
        assert _columns(conn, "training_dataset") == [
            "id", "match_id", "feature_json", "winner_label", "labels_json", "feature_extractor"]
        assert _columns(conn, "crawl_queue") == [
            "id", "puuid", "region", "game_name", "tag_line", "priority", "is_processed", "created_at", "processed_at"]
        assert _columns(conn, "match_participants")[-1] == "team_id"
        assert _columns(conn, "matches") == [
            "match_id", "raw_json", "game_mode", "game_creation_time", "game_duration", "season_id"]
        queue = {r["puuid"]: r for r in _rows(conn, "SELECT * FROM crawl_queue")}
        players = {r["puuid"]: r for r in _rows(conn, "SELECT * FROM players")}
        training = {r["id"]: r for r in _rows(conn, "SELECT * FROM training_dataset")}
        anchors = _rows(conn, "SELECT puuid, region, source, selected_at, rank_json FROM anchor_selections ORDER BY id")
        trained = _rows(conn, "SELECT * FROM trained_records")
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        orphans = [tuple(r) for r in conn.execute("PRAGMA foreign_key_check")]

    assert (queue["seed"]["created_at"], queue["seed"]["processed_at"]) == (SEED_ENTRY_MS, SEED_ENTRY_MS + 52_500_000)
    assert queue["snowball"]["id"] == 2 and queue["snowball"]["processed_at"] is None
    assert players["pro"]["updated_at"] == 1777708800000
    assert players["smurf"]["updated_at"] is None
    assert training[1]["feature_extractor"] is None  # compressed: writer unknown
    assert training[2]["feature_extractor"] == "json_utils.extract_match_features"  # carries post-game kda/gold
    assert trained == [{"training_id": 1, "trained_at": 1792497600000}]
    assert anchors == [
        {"puuid": "seed", "region": "EUW", "source": "legacy_crawl_queue", "selected_at": SEED_ENTRY_MS, "rank_json": None},
        {"puuid": "pro", "region": None, "source": "legacy_players_last_saved", "selected_at": 1777708800000, "rank_json": None},
    ]
    assert {"idx_crawl_queue_lookup", "idx_training_winner", "idx_anchor_selections_puuid",
            "idx_match_sightings_match"} <= indexes
    assert not {"idx_timelines_match", "idx_crawl_queue_puuid", "idx_training_is_trained",
                "idx_players_total_matches"} & indexes
    # Rows are copied verbatim: the alias still resolves, and the pre-existing orphan is kept, not lost.
    assert orphans == [("training_dataset", 3, "matches", 0)]


def test_rebuilt_tables_keep_their_autoincrement_sequence(tmp_path):
    db = _v15_database(tmp_path / "legacy.db")
    _legacy_rows(db)
    database.migrate_db(db)

    with database.get_connection(db) as conn:
        conn.execute("INSERT INTO crawl_queue (puuid, region) VALUES ('new', 'EUW')")
        conn.execute("INSERT INTO training_dataset (match_id) VALUES ('EUW1_3')")
        queue_id = conn.execute("SELECT id FROM crawl_queue WHERE puuid = 'new'").fetchone()[0]
        training_id = conn.execute("SELECT id FROM training_dataset WHERE match_id = 'EUW1_3'").fetchone()[0]
        created = conn.execute("SELECT created_at FROM crawl_queue WHERE puuid = 'new'").fetchone()[0]

    # Ids of deleted rows are never reused, as with the original tables.
    assert (queue_id, training_id) == (4, 10)
    assert type(created) is int and abs(created - time.time() * 1000) < 60_000


def test_a_failing_step_leaves_the_database_at_its_previous_version(tmp_path, monkeypatch):
    db = _v15_database(tmp_path / "legacy.db")
    _legacy_rows(db)
    steps = [(v, s + ["SELECT * FROM no_such_table"] if v == 15 else s) for v, s in database._MIGRATIONS]
    monkeypatch.setattr(database, "_MIGRATIONS", steps)

    with pytest.raises(sqlite3.OperationalError, match="no_such_table"):
        database.migrate_db(db)

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 15
        assert "avg_kda" in _columns(conn, "players")
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'match_facts'").fetchone() is None
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0  # plain connection default


def test_fresh_and_migrated_databases_have_the_same_schema(tmp_path):
    fresh, legacy = str(tmp_path / "fresh.db"), _v15_database(tmp_path / "legacy.db")
    database.init_db(fresh)
    database.migrate_db(legacy)

    def schema(path):
        with sqlite3.connect(path) as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            return {t: [tuple(r[1:]) for r in conn.execute(f"PRAGMA table_info({t})")] for t in tables}

    assert schema(fresh) == schema(legacy)


def _facts(db, match_id):
    with sqlite3.connect(db) as conn:
        rows = _rows(conn, f"SELECT * FROM match_facts WHERE match_id = '{match_id}'")
    return rows[0] if rows else None


def test_writers_record_match_facts_with_availability(tmp_path):
    db = str(tmp_path / "v16.db")
    database.init_db(db)
    before = int(time.time() * 1000)
    database.save_matches_batch([("EUW1_1", _payload())], db_path=db, fetched_by_puuid="puuid_7")
    after = int(time.time() * 1000)
    database.save_match("EUW1_2", _payload("EUW1_2"), db_path=db, fetched_by_puuid="stranger", ingested_at=1234)

    first = _facts(db, "EUW1_1")
    assert before <= first.pop("ingested_at") <= after
    assert first == {"match_id": "EUW1_1", "queue_id": 420, "game_start_at": 1_700_000_030_000,
                     "game_end_at": 1_700_001_830_000, "game_version": "16.18.1", "platform": "EUW1",
                     "fetched_by_puuid": "puuid_7"}
    # An anchor that did not play the match is not recorded.
    assert (_facts(db, "EUW1_2")["ingested_at"], _facts(db, "EUW1_2")["fetched_by_puuid"]) == (1234, None)
    assert database.get_raw_match("EUW1_1", db_path=db) == _payload()


def test_first_ingestion_wins_and_legacy_matches_get_no_invented_facts(tmp_path):
    db = str(tmp_path / "v16.db")
    database.init_db(db)
    with database.get_connection(db) as conn:
        conn.execute("INSERT INTO matches (match_id, game_mode) VALUES ('EUW1_LEGACY', 'CLASSIC')")
    database.save_matches_batch([("EUW1_1", _payload())], db_path=db, fetched_by_puuid="puuid_1", ingested_at=1000)
    database.save_matches_batch([("EUW1_1", _payload()), ("EUW1_LEGACY", _payload("EUW1_LEGACY"))],
                                db_path=db, fetched_by_puuid="puuid_2", ingested_at=2000)

    assert (_facts(db, "EUW1_1")["ingested_at"], _facts(db, "EUW1_1")["fetched_by_puuid"]) == (1000, "puuid_1")
    # Already stored before v16: its availability stays unknown rather than "now".
    assert _facts(db, "EUW1_LEGACY") is None


def test_participants_record_their_team(tmp_path):
    db = str(tmp_path / "v16.db")
    database.init_db(db)
    database.save_matches_batch([("EUW1_1", _payload())], db_path=db)
    database.save_match("EUW1_2", _payload("EUW1_2"), db_path=db)
    with sqlite3.connect(db) as conn:
        teams = dict(conn.execute("SELECT puuid, team_id FROM match_participants WHERE match_id = 'EUW1_2'"))
        assert conn.execute("SELECT COUNT(*) FROM match_participants WHERE team_id IS NULL").fetchone()[0] == 0
    assert teams == {f"puuid_{i}": 100 if i < 5 else 200 for i in range(10)}


def test_participant_write_errors_are_no_longer_swallowed(tmp_path):
    db = str(tmp_path / "v16.db")
    database.init_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE match_participants")
    with pytest.raises(sqlite3.OperationalError, match="match_participants"):
        database.save_matches_batch([("EUW1_1", _payload())], db_path=db)
    with pytest.raises(sqlite3.OperationalError, match="match_participants"):
        database.save_match("EUW1_2", _payload("EUW1_2"), db_path=db)
