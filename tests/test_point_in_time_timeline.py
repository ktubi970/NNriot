"""Unit tests for point-in-time timeline metrics streaming in load_enriched_records."""
import base64
import gzip
import json
from pathlib import Path
import sqlite3
import pytest

from core.feature_labels import ALL_LABEL_KEYS
from core.point_in_time_dataset import load_enriched_records


def _create_sample_timeline(p1_gold: float = 6000.0, p6_gold: float = 4500.0, fb_killer: int = 1) -> dict:
    """Create a synthetic Match-V5 timeline payload."""
    frames = [
        {
            "timestamp": 0,
            "participantFrames": {
                str(i): {"totalGold": 500, "minionsKilled": 0, "jungleMinionsKilled": 0, "xp": 0}
                for i in range(1, 11)
            },
            "events": [],
        },
        {
            "timestamp": 900000,  # 15 minutes
            "participantFrames": {
                str(i): {
                    "totalGold": p1_gold if i == 1 else (p6_gold if i == 6 else 5000.0),
                    "minionsKilled": 120 if i == 1 else (100 if i == 6 else 110),
                    "jungleMinionsKilled": 0,
                    "xp": 5500 if i == 1 else (4500 if i == 6 else 5000),
                }
                for i in range(1, 11)
            },
            "events": [
                {
                    "type": "CHAMPION_KILL",
                    "killerId": fb_killer,
                    "victimId": 6,
                    "timestamp": 180000,  # 3 minutes
                    "assistingParticipantIds": [],
                }
            ],
        },
    ]
    return {
        "info": {
            "participants": [{"participantId": i, "puuid": f"puuid_{i}"} for i in range(1, 11)],
            "frames": frames,
        }
    }


def _setup_test_db(db_path: Path, include_timelines_table: bool = True):
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
                win INTEGER,
                kills INTEGER,
                deaths INTEGER,
                assists INTEGER,
                total_cs INTEGER,
                gold_earned INTEGER,
                game_duration INTEGER
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

        if include_timelines_table:
            conn.execute("""
                CREATE TABLE match_timelines (
                    match_id TEXT PRIMARY KEY,
                    timeline_json TEXT
                )
            """)

        labels = {k: 0 for k in ALL_LABEL_KEYS}
        labels["winner"] = 1
        labels_json_str = json.dumps(labels)

        # Insert 3 consecutive matches chronologically
        for i in range(1, 4):
            mid = f"MATCH_{i}"
            gtime = 1_000_000 + i * 1_000
            conn.execute("INSERT INTO matches VALUES (?, 'CLASSIC', ?)", (mid, gtime))
            conn.execute("INSERT INTO training_dataset VALUES (?, '{}', ?, 1)", (mid, labels_json_str))

            for p in range(1, 6):
                conn.execute(
                    "INSERT INTO match_participants VALUES (?, ?, 'Aatrox', 'TOP', 1, 3, 1, 5, 150, 8000, 1800)",
                    (mid, f"puuid_{p}"),
                )
            for p in range(6, 11):
                conn.execute(
                    "INSERT INTO match_participants VALUES (?, ?, 'Zed', 'MIDDLE', 0, 1, 3, 2, 130, 6500, 1800)",
                    (mid, f"puuid_{p}"),
                )

            if include_timelines_table:
                # Store match 1 as gzipped base64
                if i == 1:
                    t1_dict = _create_sample_timeline(p1_gold=6000.0, p6_gold=4500.0, fb_killer=1)
                    comp1 = base64.b64encode(gzip.compress(json.dumps(t1_dict).encode("utf-8"))).decode("ascii")
                    conn.execute("INSERT INTO match_timelines VALUES (?, ?)", (mid, comp1))
                # Store match 2 as raw JSON string
                elif i == 2:
                    t2_dict = _create_sample_timeline(p1_gold=4000.0, p6_gold=6500.0, fb_killer=6)
                    conn.execute("INSERT INTO match_timelines VALUES (?, ?)", (mid, json.dumps(t2_dict)))
                # Match 3 has no timeline entry


def test_backward_compatibility(tmp_path):
    """When include_timeline=False (default), participant priors must contain exactly 10 floats."""
    db_file = tmp_path / "test_backward.db"
    _setup_test_db(db_file, include_timelines_table=True)

    records = load_enriched_records(str(db_file), limit=10, warmup_matches=0)
    assert len(records) == 3
    for rec in records:
        for team in ("team_a", "team_b"):
            for p in rec["feature_json"][team]:
                assert len(p["priors"]) == 10

    # Explicit include_timeline=False
    records_explicit = load_enriched_records(str(db_file), limit=10, warmup_matches=0, include_timeline=False)
    for rec in records_explicit:
        for team in ("team_a", "team_b"):
            for p in rec["feature_json"][team]:
                assert len(p["priors"]) == 10


def test_include_timeline_cold_start_and_progression(tmp_path):
    """When include_timeline=True, 14 priors are returned with strict point-in-time progression."""
    db_file = tmp_path / "test_timeline.db"
    _setup_test_db(db_file, include_timelines_table=True)

    records = load_enriched_records(str(db_file), limit=10, warmup_matches=0, include_timeline=True)
    assert len(records) == 3

    # Match 1: Cold start for timeline priors
    m1_a = records[0]["feature_json"]["team_a"]
    m1_b = records[0]["feature_json"]["team_b"]
    assert len(m1_a[0]["priors"]) == 14
    assert len(m1_b[0]["priors"]) == 14

    # Cold start timeline priors: [0.0, 0.0, 0.0, 0.10]
    p1_m1_priors = m1_a[0]["priors"]
    assert p1_m1_priors[10:] == pytest.approx([0.0, 0.0, 0.0, 0.10], abs=1e-5)

    p6_m1_priors = m1_b[0]["priors"]
    assert p6_m1_priors[10:] == pytest.approx([0.0, 0.0, 0.0, 0.10], abs=1e-5)

    # Match 2: Reflects timeline metrics from Match 1
    m2_a = records[1]["feature_json"]["team_a"]
    m2_b = records[1]["feature_json"]["team_b"]

    p1_m2_priors = m2_a[0]["priors"]
    assert len(p1_m2_priors) == 14
    # In match 1, puuid_1 had +1500 GD@15 and FBI.
    # Therefore, norm_gd15 > 0, norm_csd15 > 0, norm_xpd15 > 0, norm_fbi10 > 0.10
    assert p1_m2_priors[10] > 0.0   # norm_gd15
    assert p1_m2_priors[11] > 0.0   # norm_csd15
    assert p1_m2_priors[12] > 0.0   # norm_xpd15
    assert p1_m2_priors[13] > 0.10  # norm_fbi10

    # In match 1, puuid_6 had negative GD@15 (-1500) and no FBI
    p6_m2_priors = m2_b[0]["priors"]
    assert len(p6_m2_priors) == 14
    assert p6_m2_priors[10] < 0.0   # norm_gd15 is negative
    assert p6_m2_priors[11] < 0.0   # norm_csd15 is negative
    assert p6_m2_priors[12] < 0.0   # norm_xpd15 is negative
    assert p6_m2_priors[13] < 0.10  # norm_fbi10 < 0.10


def test_zero_future_leakage(tmp_path):
    """Ensure Match 1's timeline or Match 2's timeline never leaks into Match 1's priors."""
    db_file = tmp_path / "test_leakage.db"
    _setup_test_db(db_file, include_timelines_table=True)

    records = load_enriched_records(str(db_file), limit=10, warmup_matches=0, include_timeline=True)

    # Match 1 MUST have cold-start priors regardless of what happens in Match 1's timeline or Match 2's timeline
    p1_m1_priors = records[0]["feature_json"]["team_a"][0]["priors"]
    assert p1_m1_priors[10:] == pytest.approx([0.0, 0.0, 0.0, 0.10], abs=1e-5)


def test_missing_match_timelines_table_graceful(tmp_path):
    """When match_timelines table does not exist, include_timeline=True should return default 14 priors."""
    db_file = tmp_path / "test_no_timelines_table.db"
    _setup_test_db(db_file, include_timelines_table=False)

    records = load_enriched_records(str(db_file), limit=10, warmup_matches=0, include_timeline=True)
    assert len(records) == 3
    for rec in records:
        for team in ("team_a", "team_b"):
            for p in rec["feature_json"][team]:
                assert len(p["priors"]) == 14
                assert p["priors"][10:] == pytest.approx([0.0, 0.0, 0.0, 0.10], abs=1e-5)


def test_corrupt_timeline_graceful(tmp_path):
    """Corrupt timeline_json strings should be ignored gracefully without error."""
    db_file = tmp_path / "test_corrupt.db"
    _setup_test_db(db_file, include_timelines_table=True)

    with sqlite3.connect(db_file) as conn:
        # Overwrite MATCH_1 timeline with invalid gibberish
        conn.execute("UPDATE match_timelines SET timeline_json = 'INVALID_NOT_JSON' WHERE match_id = 'MATCH_1'")

    records = load_enriched_records(str(db_file), limit=10, warmup_matches=0, include_timeline=True)
    assert len(records) == 3
    # MATCH_2 should not have crash and puuid_1 should have cold start because MATCH_1 timeline was invalid
    p1_m2_priors = records[1]["feature_json"]["team_a"][0]["priors"]
    assert p1_m2_priors[10:] == pytest.approx([0.0, 0.0, 0.0, 0.10], abs=1e-5)
