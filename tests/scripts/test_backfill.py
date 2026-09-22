import pytest

pytestmark = pytest.mark.integration


import json
import sqlite3
from core import database

def test_backfill_handles_corrupt_or_empty_match_without_loop(tmp_path):
    db_path = str(tmp_path / "test_corrupt.db")
    database.init_db(db_path)

    # Insert a corrupt match with invalid JSON and an empty match
    with database.get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO matches (match_id, raw_json, game_mode) VALUES (?, ?, ?)",
            ("CORRUPT_1", "INVALID_BASE64_OR_NOT_JSON", "CLASSIC"),
        )
        conn.execute(
            "INSERT INTO matches (match_id, raw_json, game_mode) VALUES (?, ?, ?)",
            ("EMPTY_1", "", "CLASSIC"),
        )

    # Run backfill: should process 2 matches and not loop
    count1 = database.backfill_match_participants(limit=10, db_path=db_path)
    assert count1 == 2

    # Second run must find 0 unbackfilled matches
    count2 = database.backfill_match_participants(limit=10, db_path=db_path)
    assert count2 == 0


def test_backfill_handles_null_info_and_malformed_participants(tmp_path):
    db_path = str(tmp_path / "test_null_info.db")
    database.init_db(db_path)

    with database.get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO matches (match_id, raw_json, game_mode) VALUES (?, ?, ?)",
            ("NULL_INFO_1", json.dumps({"info": None}), "CLASSIC"),
        )
        conn.execute(
            "INSERT INTO matches (match_id, raw_json, game_mode) VALUES (?, ?, ?)",
            ("MALFORMED_PARTICIPANT_1", json.dumps({"info": {"participants": ["malformed"]}}), "CLASSIC"),
        )

    count1 = database.backfill_match_participants(limit=10, db_path=db_path)
    assert count1 == 2

    # Verify match_participants contains the tombstone rows
    with database.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT match_id, puuid FROM match_participants WHERE match_id IN (?, ?)",
            ("NULL_INFO_1", "MALFORMED_PARTICIPANT_1"),
        ).fetchall()
        tombstones = {r["match_id"]: r["puuid"] for r in rows}
        assert tombstones.get("NULL_INFO_1") == "__TOMBSTONE__"
        assert tombstones.get("MALFORMED_PARTICIPANT_1") == "__TOMBSTONE__"

    # Second run must find 0 unbackfilled matches
    count2 = database.backfill_match_participants(limit=10, db_path=db_path)
    assert count2 == 0
