import json
from pathlib import Path
import pytest
from scripts.benchmark_draft_signal import run_draft_signal_benchmark, main


def make_biased_record(mid: int, bias: float = 0.8, force_winner: int | None = None):
    # Deterministic generation where champion pick drives winner
    is_darius = (mid % 2 == 0)
    if force_winner is not None:
        winner = force_winner
    else:
        winner = 1 if (is_darius and (mid % 10 < int(bias * 10))) else 0

    full_labels = {
        "winner": winner,
        "team_b_kill_lead": 0,
        "kill_handicap": 0,
        "total_kills": 20,
        "team_a_kills": 10,
        "team_b_kills": 10,
        "kills_odd": 0,
        "first_blood": 0,
        "first_baron": 0,
        "first_inhibitor": 0,
        "first_tower": 0,
        "total_barons": 0,
        "total_dragons": 0,
        "total_towers": 0,
        "both_baron": 0,
        "both_inhibitor": 0,
        "both_dragon": 0,
        "elder_dragon": 0,
    }

    return {
        "match_id": f"BENCH_{mid}",
        "game_creation_time": 1700000000 + mid * 10,
        "feature_json": {
            "team_a": [
                {"champion": "Aatrox", "role": "TOP"},
                {"champion": "LeeSin", "role": "JUNGLE"},
                {"champion": "Ahri", "role": "MIDDLE"},
                {"champion": "Jinx", "role": "BOTTOM"},
                {"champion": "Thresh", "role": "UTILITY"},
            ],
            "team_b": [
                {"champion": "Darius" if is_darius else "Garen", "role": "TOP"},
                {"champion": "Elise", "role": "JUNGLE"},
                {"champion": "Zed", "role": "MIDDLE"},
                {"champion": "Ezreal", "role": "BOTTOM"},
                {"champion": "Lulu", "role": "UTILITY"},
            ],
            "game_mode": "CLASSIC",
        },
        "labels_json": full_labels,
        "winner_label": winner,
    }


def test_benchmark_detects_signal_when_present():
    records = [make_biased_record(i, bias=0.85) for i in range(100)]
    result = run_draft_signal_benchmark(records, min_samples=60)
    assert "verdict" in result
    assert "test_auc" in result
    assert "auc_ci_95" in result
    assert result["auc_ci_95"][0] <= result["test_auc"] <= result["auc_ci_95"][1]
    assert result["verdict"] == "SIGNAL_CONFIRMED"
    assert result["test_auc"] > 0.60
    assert result["p_value_vs_null"] < 0.05
    assert result["sample_counts"]["train"] + result["sample_counts"]["validation"] + result["sample_counts"]["test"] == 100


def test_benchmark_raises_on_insufficient_samples():
    records = [make_biased_record(i) for i in range(30)]
    with pytest.raises(ValueError, match="Need at least 50 records"):
        run_draft_signal_benchmark(records, min_samples=50)


def test_benchmark_handles_zero_division_when_single_class():
    # Construct 100 records where training has both classes, but test set has only class 0
    # Chronological split puts the last ~15% in test (e.g. index 85 to 99)
    records = []
    for i in range(100):
        if i >= 85:
            records.append(make_biased_record(i, force_winner=0))
        else:
            records.append(make_biased_record(i, bias=0.5))

    result = run_draft_signal_benchmark(records, min_samples=60)
    assert result["verdict"] == "NO_SIGNAL"
    assert result["p_value_vs_null"] == 1.0
    assert result["auc_standard_error"] == 1.0


def test_benchmark_cli_main(tmp_path, monkeypatch):
    import sqlite3

    db_path = tmp_path / "test.db"
    out_path = tmp_path / "report.json"

    # Create dummy database
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE matches (
                match_id TEXT PRIMARY KEY,
                game_creation_time INTEGER,
                game_mode TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE training_dataset (
                id INTEGER PRIMARY KEY,
                match_id TEXT,
                feature_json TEXT,
                labels_json TEXT,
                winner_label INTEGER
            )
        """)
        for i in range(100):
            rec = make_biased_record(i, bias=0.8)
            conn.execute(
                "INSERT INTO matches VALUES (?, ?, ?)",
                (rec["match_id"], rec["game_creation_time"], "CLASSIC"),
            )
            conn.execute(
                "INSERT INTO training_dataset VALUES (?, ?, ?, ?, ?)",
                (
                    i + 1,
                    rec["match_id"],
                    json.dumps(rec["feature_json"]),
                    json.dumps(rec["labels_json"]),
                    rec["winner_label"],
                ),
            )

    main([
        "--db",
        str(db_path),
        "--samples",
        "100",
        "--output",
        str(out_path),
    ])

    assert out_path.exists()
    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "verdict" in data
    assert "test_auc" in data
