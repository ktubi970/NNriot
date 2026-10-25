"""Backfill of match_facts and participant teams for matches stored before schema v16."""
import base64
import gzip
import json
import sqlite3

from core import database
from scripts.backfill_match_facts import backfill


def _payload(match_id, creation=1000):
    parts = [{"puuid": f"{match_id}-p{i}", "teamId": 100 if i < 5 else 200, "championName": "Ahri",
              "win": i < 5} for i in range(10)]
    return {"metadata": {"matchId": match_id}, "info": {
        "gameMode": "CLASSIC", "queueId": 440, "platformId": "KR", "gameCreation": creation,
        "gameStartTimestamp": creation + 10, "gameEndTimestamp": creation + 2000,
        "gameVersion": "15.3.1", "participants": parts}}


def _legacy_db(tmp_path):
    path = str(tmp_path / "legacy.db")
    with sqlite3.connect(path) as conn:
        for from_version, statements in database._MIGRATIONS:
            if from_version >= 15:
                break
            for sql in statements:
                conn.execute(sql)
        compressed = base64.b64encode(gzip.compress(json.dumps(_payload("KR_1")).encode())).decode()
        conn.executemany("INSERT INTO matches (match_id, raw_json, game_mode) VALUES (?, ?, 'CLASSIC')",
                         [("KR_1", compressed), ("KR_2", json.dumps(_payload("KR_2"))), ("KR_BAD", "not-json")])
        conn.executemany("INSERT INTO match_participants (match_id, puuid) VALUES (?, ?)",
                         [("KR_1", f"KR_1-p{i}") for i in range(10)])
    database.migrate_db(path)
    return path


def _facts(path):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return {r["match_id"]: dict(r) for r in conn.execute("SELECT * FROM match_facts")}


def test_legacy_matches_get_derived_facts_with_unknown_availability(tmp_path):
    path = _legacy_db(tmp_path)
    database.save_matches_batch([("KR_NEW", _payload("KR_NEW", 5000))], db_path=path,
                                fetched_by_puuid="KR_NEW-p0", ingested_at=9000)
    written = _facts(path)["KR_NEW"]

    report = backfill(path, batch_size=1)

    facts = _facts(path)
    assert report == {"facts_inserted": 3, "participants_updated": 10, "batches": 3}
    assert facts["KR_1"] == {"match_id": "KR_1", "queue_id": 440, "game_start_at": 1010, "game_end_at": 3000,
                             "game_version": "15.3.1", "platform": "KR", "ingested_at": None,
                             "fetched_by_puuid": None}
    assert facts["KR_2"]["queue_id"] == 440
    # An unreadable payload still gets a row, so it is not scanned again.
    assert facts["KR_BAD"] == {"match_id": "KR_BAD", "queue_id": None, "game_start_at": None, "game_end_at": None,
                               "game_version": None, "platform": "KR", "ingested_at": None, "fetched_by_puuid": None}
    assert facts["KR_NEW"] == written
    with sqlite3.connect(path) as conn:
        teams = dict(conn.execute("SELECT puuid, team_id FROM match_participants WHERE match_id = 'KR_1'"))
    assert teams == {f"KR_1-p{i}": 100 if i < 5 else 200 for i in range(10)}


def test_backfill_is_resumable_and_bounded(tmp_path):
    path = _legacy_db(tmp_path)
    assert backfill(path, batch_size=1, max_batches=2)["facts_inserted"] == 2
    assert backfill(path, batch_size=10) == {"facts_inserted": 1, "participants_updated": 0, "batches": 1}
    assert backfill(path) == {"facts_inserted": 0, "participants_updated": 0, "batches": 0}
