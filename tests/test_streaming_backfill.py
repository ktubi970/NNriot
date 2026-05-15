"""Test that backfill_labels processes rows in chunks without loading them all."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database
import backfill_labels as bl
from data_collector import _build_training_record


def _make_classic_match(match_id, team_a_win=True):
    return {
        "info": {
            "gameMode": "CLASSIC",
            "gameVersion": "14.10.400",
            "participants": [
                {"championName": "Aatrox", "teamId": 100, "kills": 5, "deaths": 3, "assists": 7, "goldEarned": 12000, "teamPosition": "TOP"},
                {"championName": "Lux",    "teamId": 200, "kills": 4, "deaths": 6, "assists": 8, "goldEarned": 11000, "teamPosition": "MID"},
            ],
            "teams": [
                {"teamId": 100, "win": team_a_win, "objectives": {}},
                {"teamId": 200, "win": not team_a_win, "objectives": {}},
            ],
        }
    }


def test_backfill_chunks_through_many_rows(tmp_path):
    """Insert 25 rows with NULL labels, backfill with chunk_size=5, verify all populated."""
    db = str(tmp_path / "test.db")
    database.init_db(db)

    # Insert 25 matches + corresponding 3-tuple training records (no labels)
    matches = []
    records = []
    for i in range(25):
        m_id = f"M_{i:03d}"
        match = _make_classic_match(m_id, team_a_win=(i % 2 == 0))
        matches.append((m_id, match))
        rec = _build_training_record(m_id, match)
        assert rec is not None
        records.append(rec)  # 3-tuple — no labels

    database.save_matches_batch(matches, db_path=db)
    database.save_training_records_batch(records, db_path=db)

    # All 25 should have NULL labels_json
    with database.get_connection(db) as conn:
        n_null = conn.execute("SELECT COUNT(*) FROM training_dataset WHERE labels_json IS NULL").fetchone()[0]
    assert n_null == 25

    # Backfill with a small chunk to force multiple iterations
    stats = bl.backfill_labels(db_path=db, chunk_size=5)
    assert stats["total_seen"] == 25
    assert stats["updated"] == 25
    assert stats["skipped_malformed"] == 0
    assert stats["errors"] == 0

    with database.get_connection(db) as conn:
        n_null_after = conn.execute("SELECT COUNT(*) FROM training_dataset WHERE labels_json IS NULL").fetchone()[0]
    assert n_null_after == 0


def test_backfill_idempotent(tmp_path):
    """Running backfill twice doesn't re-process anything."""
    db = str(tmp_path / "test.db")
    database.init_db(db)

    match = _make_classic_match("M_idem")
    database.save_matches_batch([("M_idem", match)], db_path=db)
    rec = _build_training_record("M_idem", match)
    database.save_training_records_batch([rec], db_path=db)

    stats1 = bl.backfill_labels(db_path=db, chunk_size=10)
    assert stats1["updated"] == 1

    stats2 = bl.backfill_labels(db_path=db, chunk_size=10)
    assert stats2["total_seen"] == 0
    assert stats2["updated"] == 0


def test_backfill_breaks_on_all_malformed(tmp_path):
    """If all NULL rows are malformed, backfill breaks out instead of looping forever."""
    db = str(tmp_path / "test.db")
    database.init_db(db)

    # Insert a CHERRY match (will produce a record because _build_training_record
    # has the EXCLUDED filter — but we bypass it for this test).
    # Manually insert a malformed match: missing team 200
    bad_match = {
        "info": {
            "gameMode": "CLASSIC",
            "gameVersion": "14.10.400",
            "participants": [
                {"championName": "Aatrox", "teamId": 100, "kills": 5, "deaths": 3, "assists": 7, "goldEarned": 12000, "teamPosition": "TOP"},
            ],
            "teams": [
                {"teamId": 100, "win": True, "objectives": {}},
            ],
        }
    }
    database.save_matches_batch([("M_bad", bad_match)], db_path=db)
    # Insert training_dataset row directly bypassing _build_training_record's checks
    import json as _json, gzip, base64
    fj = base64.b64encode(gzip.compress(_json.dumps({"team_a": [], "team_b": []}).encode())).decode("ascii")
    with database.get_connection(db) as conn:
        conn.execute(
            "INSERT INTO training_dataset (match_id, feature_json, winner_label, labels_json) VALUES (?, ?, ?, NULL)",
            ("M_bad", fj, 0)
        )

    stats = bl.backfill_labels(db_path=db, chunk_size=5)
    # Should have seen the row, marked it malformed, and exited cleanly
    assert stats["updated"] == 0
    assert stats["skipped_malformed"] == 1


def test_backfill_breaks_on_mixed_then_malformed_tail(tmp_path):
    """
    Mixed-then-tail case: 3 good + 3 malformed rows, chunk_size=2.
    The streaming loop must terminate after exhausting good rows and seeing a
    chunk of pure malformed rows. Without the per-chunk safety break this
    case loops forever.
    """
    db = str(tmp_path / "test.db")
    database.init_db(db)

    # 3 good matches
    matches = []
    records = []
    for i in range(3):
        m_id = f"M_good_{i}"
        match = _make_classic_match(m_id, team_a_win=(i % 2 == 0))
        matches.append((m_id, match))
        rec = _build_training_record(m_id, match)
        assert rec is not None
        records.append(rec)
    database.save_matches_batch(matches, db_path=db)
    database.save_training_records_batch(records, db_path=db)

    # 3 malformed (missing team 200) — bypass _build_training_record's filter
    import json as _json, gzip, base64
    for i in range(3):
        m_id = f"M_bad_{i}"
        # Match with only team 100 — extract_labels will return None
        bad_match = {
            "info": {
                "gameMode": "CLASSIC",
                "gameVersion": "14.10.400",
                "participants": [{"championName": "Aatrox", "teamId": 100, "kills": 5, "deaths": 3, "assists": 7, "goldEarned": 12000, "teamPosition": "TOP"}],
                "teams": [{"teamId": 100, "win": True, "objectives": {}}],
            }
        }
        database.save_matches_batch([(m_id, bad_match)], db_path=db)
        fj = base64.b64encode(gzip.compress(_json.dumps({}).encode())).decode("ascii")
        with database.get_connection(db) as conn:
            conn.execute(
                "INSERT INTO training_dataset (match_id, feature_json, winner_label, labels_json) VALUES (?, ?, ?, NULL)",
                (m_id, fj, 0)
            )

    # 3 good + 3 bad, chunk_size=2 → at least 3 iterations before the all-malformed chunk.
    # Without the per-chunk break, this hangs.
    import threading
    result = {}
    def run():
        result["stats"] = bl.backfill_labels(db_path=db, chunk_size=2)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "backfill_labels did not terminate within 15s — infinite loop"
    stats = result["stats"]
    assert stats["updated"] == 3
    assert stats["total_seen"] >= 3  # at minimum saw the 3 good ones plus some malformed
