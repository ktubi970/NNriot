"""Guards of the read-only crawl seed bias diagnostic: sealed test untouched, cutoff and orientation."""

import json
import sqlite3

import pytest

from core import database
from core.collection.provenance import CrawlRegistry, match_provenance
from scripts.diagnose_crawl_seed_bias import (
    add_oriented,
    dataset_report,
    load_split_rows,
    player_category,
    rates_table,
    read_db_teams,
    sealed_test_cutoff_ms,
)


def _row(match_id, creation, team_a, team_b, winner):
    return {"match_id": match_id, "game_creation_time": creation, "labels_json": {"winner": winner},
            "feature_json": {"team_a": [{"puuid": p} for p in team_a], "team_b": [{"puuid": p} for p in team_b]}}


@pytest.fixture
def dataset_dir(tmp_path):
    (tmp_path / "train.json").write_text(json.dumps([_row("T1", 10, ["a"], ["b"], 1)]), encoding="utf-8")
    (tmp_path / "validation.json").write_text(json.dumps([_row("V1", 20, ["c"], ["d"], 0)]), encoding="utf-8")
    (tmp_path / "model.split.json").write_text(json.dumps({"splits": {
        "train": [], "validation": [],
        "test": [{"match_id": "S2", "game_creation_time": 40}, {"match_id": "S1", "game_creation_time": 30}]}}),
        encoding="utf-8")
    # Unreadable on purpose: the diagnostic must never open the sealed test.
    (tmp_path / "sealed_test.json").write_bytes(b"\x00not json")
    return tmp_path


def test_split_rows_come_from_train_and_validation_only(dataset_dir):
    rows = list(load_split_rows(dataset_dir))
    assert [(r["split"], r["match_id"], r["team_b"], r["winner"]) for r in rows] == [
        ("train", "T1", ["b"], 1), ("validation", "V1", ["d"], 0)]


def test_cutoff_is_the_first_sealed_test_creation(dataset_dir):
    assert sealed_test_cutoff_ms(dataset_dir) == 30


def test_artifact_side_is_oriented_on_the_seed_team():
    registry = CrawlRegistry(seed_entry_ms={"seed": 100}, anchor_saved_ms={}, crawled=frozenset({"seed"}))
    table = rates_table()
    prov = match_provenance(["seed"], ["x"], 50, registry)
    add_oriented(table, prov, b_won=True, periods=("p",))
    assert (table["ladder_seed_team"]["p"].n, table["ladder_seed_team"]["p"].k) == (1, 0)
    assert table["pre_entry_seed_team"]["all"].k == 0
    assert "post_entry_only_seed_team" not in table


def test_crawled_imbalance_is_split_by_crawl_time():
    registry = CrawlRegistry(seed_entry_ms={}, anchor_saved_ms={}, crawled=frozenset({"early", "late"}),
                             crawled_at_ms={"early": 10, "late": 90})
    table = rates_table()
    add_oriented(table, match_provenance(["x"], ["late"], 50, registry), b_won=True, periods=("p",))
    add_oriented(table, match_provenance(["early"], ["x"], 50, registry), b_won=True, periods=("p",))
    assert (table["more_crawled_after_team_no_pre_entry_anchor"]["all"].n,
            table["more_crawled_after_team_no_pre_entry_anchor"]["all"].k) == (1, 1)
    assert (table["more_crawled_before_team_no_pre_entry_anchor"]["all"].n,
            table["more_crawled_before_team_no_pre_entry_anchor"]["all"].k) == (1, 0)
    # Dose-response: matches are also grouped by the size of the gap, capped at 3.
    assert table["more_crawled_team_no_pre_entry_anchor_gap1"]["all"].n == 2


def test_dataset_report_counts_each_match_once_in_all():
    registry = CrawlRegistry(seed_entry_ms={"seed": 100}, anchor_saved_ms={}, crawled=frozenset())
    rows = [{"split": "train", "match_id": f"M{i}", "game_creation_time": 50, "team_a": ["seed"],
             "team_b": ["x"], "winner": i % 2} for i in range(3)]
    report, companion = dataset_report(rows, registry)
    assert report["artifact_side_wins"]["ladder_seed_team"]["all"]["n"] == 3
    assert report["artifact_side_wins"]["ladder_seed_team"]["train"]["n"] == 3
    assert len(companion) == 3


def test_player_category_precedence():
    registry = CrawlRegistry(seed_entry_ms={"s": 100}, anchor_saved_ms={"a": 100},
                             crawled=frozenset({"s", "a", "c"}))
    queued = frozenset({"s", "a", "c", "q"})
    got = [player_category(p, 50, registry, queued) for p in ("s", "a", "c", "q", "n")]
    assert got == ["seed_pre_entry", "anchor_pre_save", "crawled", "queued_pending", "never_queued"]
    assert player_category("s", 100, registry, queued) == "seed_post_entry"


def test_db_teams_stop_before_cutoff_and_skip_remakes(tmp_path):
    db_file = str(tmp_path / "teams.db")
    database.init_db(db_file)
    rows = []
    for match_id, creation, duration in (("OK", 10, 1800), ("LATE", 30, 1800), ("REMAKE", 10, 200)):
        rows += [(match_id, f"{match_id}{i}", int(i < 5), creation, duration, "CLASSIC") for i in range(10)]
    with sqlite3.connect(db_file) as conn:
        conn.executemany("INSERT INTO matches (match_id) VALUES (?)", [("OK",), ("LATE",), ("REMAKE",)])
        conn.executemany(
            "INSERT INTO match_participants (match_id, puuid, win, game_creation_time, game_duration, game_mode) "
            "VALUES (?, ?, ?, ?, ?, ?)", rows)
    teams = list(read_db_teams(db_file, cutoff_ms=30, min_duration_s=300))
    assert teams == [("OK", 10, [f"OK{i}" for i in range(5)], [f"OK{i}" for i in range(5, 10)])]
