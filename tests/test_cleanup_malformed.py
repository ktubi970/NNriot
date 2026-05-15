"""Test the malformed-row cleanup script."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database
import cleanup_malformed as cm
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


def test_dry_run_reports_count_without_deleting(tmp_path):
    """Dry run mode reports the count but doesn't delete."""
    db = str(tmp_path / "test.db")
    database.init_db(db)

    # Mix: 2 good records (with labels), 2 bad (NULL labels)
    for i in range(2):
        m_id = f"M_good_{i}"
        match = _make_classic_match(m_id)
        database.save_matches_batch([(m_id, match)], db_path=db)
        rec = _build_training_record(m_id, match)
        from feature_labels import extract_labels
        labels = extract_labels(match)
        database.save_training_records_batch([rec + (labels,)], db_path=db)

    import json as _json, gzip, base64
    for i in range(2):
        m_id = f"M_bad_{i}"
        match = _make_classic_match(m_id)  # OK match for matches table
        database.save_matches_batch([(m_id, match)], db_path=db)
        fj = base64.b64encode(gzip.compress(_json.dumps({}).encode())).decode("ascii")
        with database.get_connection(db) as conn:
            conn.execute(
                "INSERT INTO training_dataset (match_id, feature_json, winner_label, labels_json) VALUES (?, ?, ?, NULL)",
                (m_id, fj, 0)
            )

    result = cm.cleanup_malformed(db_path=db, dry_run=True)
    assert result == {"would_delete": 2}

    # Verify nothing was actually deleted
    with database.get_connection(db) as conn:
        n_total = conn.execute("SELECT COUNT(*) FROM training_dataset").fetchone()[0]
    assert n_total == 4


def test_apply_deletes_only_null_labels_rows(tmp_path):
    """With --apply (dry_run=False), only NULL labels_json rows are deleted."""
    db = str(tmp_path / "test.db")
    database.init_db(db)

    import json as _json, gzip, base64
    # 1 good record
    m_id = "M_good"
    match = _make_classic_match(m_id)
    database.save_matches_batch([(m_id, match)], db_path=db)
    rec = _build_training_record(m_id, match)
    from feature_labels import extract_labels
    labels = extract_labels(match)
    database.save_training_records_batch([rec + (labels,)], db_path=db)

    # 3 bad records (NULL labels_json)
    for i in range(3):
        m_id = f"M_bad_{i}"
        match = _make_classic_match(m_id)
        database.save_matches_batch([(m_id, match)], db_path=db)
        fj = base64.b64encode(gzip.compress(_json.dumps({}).encode())).decode("ascii")
        with database.get_connection(db) as conn:
            conn.execute(
                "INSERT INTO training_dataset (match_id, feature_json, winner_label, labels_json) VALUES (?, ?, ?, NULL)",
                (m_id, fj, 0)
            )

    result = cm.cleanup_malformed(db_path=db, dry_run=False)
    assert result == {"deleted": 3}

    with database.get_connection(db) as conn:
        n_remaining = conn.execute("SELECT COUNT(*) FROM training_dataset").fetchone()[0]
        n_null = conn.execute("SELECT COUNT(*) FROM training_dataset WHERE labels_json IS NULL").fetchone()[0]
    assert n_remaining == 1
    assert n_null == 0
