"""Unit tests for point-in-time dataset streaming."""
import sqlite3
from pathlib import Path
import pytest

from core.point_in_time_dataset import load_enriched_records
from core.feature_labels import ALL_LABEL_KEYS


def _init_test_db(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE matches (
                match_id TEXT PRIMARY KEY,
                game_mode TEXT,
                game_creation_time INTEGER
            )
        """)
        conn.execute("CREATE INDEX idx_matches_game_creation ON matches(game_creation_time)")
        conn.execute("""
            CREATE TABLE match_participants (
                match_id TEXT,
                puuid TEXT,
                champion_name TEXT,
                team_position TEXT,
                win INTEGER
            )
        """)
        conn.execute("CREATE INDEX idx_match_participants_match ON match_participants(match_id)")
        conn.execute("""
            CREATE TABLE training_dataset (
                match_id TEXT PRIMARY KEY,
                feature_json TEXT,
                labels_json TEXT,
                winner_label INTEGER
            )
        """)

        labels = {k: 0 for k in ALL_LABEL_KEYS}
        labels["winner"] = 1
        labels["team_b_kill_lead"] = 0
        import json
        labels_json_str = json.dumps(labels)

        # Insert 5 consecutive matches chronologically
        for i in range(1, 6):
            mid = f"MATCH_{i}"
            gtime = 1000000 + i * 1000
            conn.execute("INSERT INTO matches VALUES (?, 'CLASSIC', ?)", (mid, gtime))
            conn.execute("INSERT INTO training_dataset VALUES (?, '{}', ?, 1)", (mid, labels_json_str))
            
            # Team A (0..4), Team B (5..9)
            for p in range(5):
                conn.execute(
                    "INSERT INTO match_participants VALUES (?, ?, 'Aatrox', 'TOP', 1)",
                    (mid, f"player_a_{p}"),
                )
            for p in range(5):
                conn.execute(
                    "INSERT INTO match_participants VALUES (?, ?, 'Zed', 'MIDDLE', 0)",
                    (mid, f"player_b_{p}"),
                )


def test_load_enriched_records_streaming_and_leak_free(tmp_path):
    db_file = tmp_path / "test.db"
    _init_test_db(db_file)

    # With warmup=2, we should get matches 3, 4, 5 (3 matches total)
    records = load_enriched_records(str(db_file), limit=10, warmup_matches=2)
    assert len(records) == 3
    assert records[0]["match_id"] == "MATCH_3"

    # Verify priors are attached to team_a participants
    feat = records[0]["feature_json"]
    assert "team_a" in feat
    assert len(feat["team_a"]) == 5
    p0 = feat["team_a"][0]
    assert "priors" in p0
    assert len(p0["priors"]) == 6
    # In match 3, player_a_0 already won match 1 and match 2
    # So their smoothed winrate should be strictly > 0.50
    assert p0["priors"][0] > 0.50
    # Champion games = 2
    assert p0["priors"][2] > 0.0


def test_load_enriched_records_validation():
    with pytest.raises(ValueError):
        load_enriched_records("nonexistent.db", limit=-1)
    with pytest.raises(ValueError):
        load_enriched_records("nonexistent.db", limit=10, warmup_matches=-1)
    with pytest.raises(FileNotFoundError):
        load_enriched_records("nonexistent.db", limit=10)
