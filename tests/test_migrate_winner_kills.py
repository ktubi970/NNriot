"""Test the winner_kills → team_b_kill_lead migration."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import gzip
import base64
import database
import migrate_winner_kills as m


def _make_match(match_id):
    return {
        "info": {
            "gameMode": "CLASSIC",
            "gameVersion": "14.10.400",
            "participants": [
                {"championName": "Aatrox", "teamId": 100, "kills": 5, "deaths": 3, "assists": 7, "goldEarned": 12000, "teamPosition": "TOP"},
                {"championName": "Lux",    "teamId": 200, "kills": 4, "deaths": 6, "assists": 8, "goldEarned": 11000, "teamPosition": "MID"},
            ],
            "teams": [
                {"teamId": 100, "win": True, "objectives": {}},
                {"teamId": 200, "win": False, "objectives": {}},
            ],
        }
    }


def _save_legacy_label_row(db, match_id, legacy_labels):
    """Insert a training_dataset row with the OLD `winner_kills` key encoded."""
    database.save_matches_batch([(match_id, _make_match(match_id))], db_path=db)
    fj = base64.b64encode(gzip.compress(b'{}')).decode("ascii")
    lj = base64.b64encode(gzip.compress(json.dumps(legacy_labels).encode())).decode("ascii")
    with database.get_connection(db) as conn:
        conn.execute(
            "INSERT INTO training_dataset (match_id, feature_json, winner_label, labels_json) VALUES (?, ?, ?, ?)",
            (match_id, fj, 0, lj)
        )


def test_dry_run_counts_without_changing(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)
    _save_legacy_label_row(db, "M1", {"winner_kills": 1, "winner": 0})

    stats = m.migrate(db_path=db, dry_run=True)
    assert stats["renamed"] == 1
    assert stats["errors"] == 0

    # Verify the row is unchanged
    with database.get_connection(db) as conn:
        row = conn.execute("SELECT labels_json FROM training_dataset").fetchone()
    labels = m._decode(row["labels_json"])
    assert "winner_kills" in labels  # untouched
    assert "team_b_kill_lead" not in labels


def test_apply_renames_key(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)
    _save_legacy_label_row(db, "M1", {"winner_kills": 1, "winner": 0, "total_kills": 30})

    stats = m.migrate(db_path=db, dry_run=False)
    assert stats["renamed"] == 1

    with database.get_connection(db) as conn:
        row = conn.execute("SELECT labels_json FROM training_dataset").fetchone()
    labels = m._decode(row["labels_json"])
    assert "winner_kills" not in labels
    assert labels["team_b_kill_lead"] == 1
    assert labels["winner"] == 0
    assert labels["total_kills"] == 30  # other keys preserved


def test_idempotent_on_already_migrated(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)
    _save_legacy_label_row(db, "M1", {"team_b_kill_lead": 1, "winner": 0})  # already migrated

    stats = m.migrate(db_path=db, dry_run=False)
    assert stats["renamed"] == 0
    assert stats["already_new"] == 1


def test_migrate_preserves_new_key_if_both_present(tmp_path):
    """When both winner_kills AND team_b_kill_lead exist, keep team_b_kill_lead value."""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    # Insert a row with BOTH keys — team_b_kill_lead=1 should survive
    _save_legacy_label_row(db, "M_dual", {"winner_kills": 0, "team_b_kill_lead": 1, "winner": 0})

    stats = m.migrate(db_path=db, dry_run=False)
    assert stats["renamed"] == 1

    with database.get_connection(db) as conn:
        row = conn.execute("SELECT labels_json FROM training_dataset").fetchone()
    labels = m._decode(row["labels_json"])
    assert "winner_kills" not in labels
    assert labels["team_b_kill_lead"] == 1  # NOT 0 from the legacy key
