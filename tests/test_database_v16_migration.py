"""Test the v15 -> v16 migration: match ingestion time and fetching anchor."""

import sqlite3
import time

from core import database


def _payload(match_id="EUW1_1", creation=1_700_000_000_000):
    participants = [{"puuid": f"puuid_{i}", "teamId": 100 if i < 5 else 200,
                     "championName": "Ahri", "win": i < 5} for i in range(10)]
    return {"metadata": {"matchId": match_id}, "info": {
        "gameMode": "CLASSIC", "gameCreation": creation, "gameDuration": 1500,
        "gameVersion": "16.18.1", "participants": participants}}


def _v15_database(path):
    with sqlite3.connect(path) as conn:
        for from_version, statements in database._MIGRATIONS:
            if from_version >= 15:
                break
            for sql in statements:
                conn.execute(sql)
    return str(path)


def _match_row(db, match_id):
    with database.get_connection(db) as conn:
        return dict(conn.execute(
            "SELECT ingested_at, fetched_by_puuid FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone())


def test_v16_adds_ingestion_columns_and_leaves_legacy_rows_unknown(tmp_path):
    db = _v15_database(tmp_path / "legacy.db")
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO matches (match_id, raw_json, game_mode, game_creation_time) "
                     "VALUES ('EUW1_OLD', NULL, 'CLASSIC', 1000)")

    database.migrate_db(db)

    with database.get_connection(db) as conn:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        columns = {r["name"]: r["type"] for r in conn.execute("PRAGMA table_info(matches)")}
    assert version == database.SCHEMA_VERSION == 16
    assert columns["ingested_at"] == "INTEGER"
    assert columns["fetched_by_puuid"] == "TEXT"
    assert _match_row(db, "EUW1_OLD") == {"ingested_at": None, "fetched_by_puuid": None}


def test_batch_writer_records_ingestion_time_and_fetching_participant(tmp_path):
    db = str(tmp_path / "v16.db")
    database.init_db(db)
    before = int(time.time() * 1000)
    database.save_matches_batch([("EUW1_1", _payload())], db_path=db, fetched_by_puuid="puuid_7")
    after = int(time.time() * 1000)

    row = _match_row(db, "EUW1_1")
    assert before <= row["ingested_at"] <= after
    assert row["fetched_by_puuid"] == "puuid_7"
    # The payload accessor read by external consumers is unchanged.
    assert database.get_raw_match("EUW1_1", db_path=db) == _payload()


def test_single_writer_records_ingestion_time(tmp_path):
    db = str(tmp_path / "v16.db")
    database.init_db(db)
    database.save_match("EUW1_1", _payload(), db_path=db, fetched_by_puuid="puuid_0", ingested_at=1234)
    database.save_match("EUW1_2", _payload("EUW1_2"), db_path=db)

    assert _match_row(db, "EUW1_1") == {"ingested_at": 1234, "fetched_by_puuid": "puuid_0"}
    assert _match_row(db, "EUW1_2")["ingested_at"] > 1234
    assert _match_row(db, "EUW1_2")["fetched_by_puuid"] is None


def test_anchor_that_did_not_play_the_match_is_not_recorded(tmp_path):
    db = str(tmp_path / "v16.db")
    database.init_db(db)
    database.save_matches_batch([("EUW1_1", _payload())], db_path=db, fetched_by_puuid="stranger")
    database.save_match("EUW1_2", _payload("EUW1_2"), db_path=db, fetched_by_puuid="stranger")

    assert _match_row(db, "EUW1_1")["fetched_by_puuid"] is None
    assert _match_row(db, "EUW1_2")["fetched_by_puuid"] is None


def test_first_ingestion_wins_when_a_match_is_fetched_again(tmp_path):
    # Availability starts at the first insert; a later fetch by another anchor must not move it.
    db = str(tmp_path / "v16.db")
    database.init_db(db)
    database.save_matches_batch([("EUW1_1", _payload())], db_path=db,
                                fetched_by_puuid="puuid_1", ingested_at=1000)
    database.save_matches_batch([("EUW1_1", _payload())], db_path=db,
                                fetched_by_puuid="puuid_2", ingested_at=2000)

    assert _match_row(db, "EUW1_1") == {"ingested_at": 1000, "fetched_by_puuid": "puuid_1"}
