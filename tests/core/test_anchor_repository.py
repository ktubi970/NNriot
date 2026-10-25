"""AnchorRepository: an append-only record of who the collector anchored on."""
import json
import sqlite3
import time

import pytest

from core import database
from core.collection.anchors import RANK_SELECTION_SOURCES, AnchorRepository


@pytest.fixture
def repo(tmp_path):
    path = str(tmp_path / "anchors.db")
    database.init_db(path)
    return AnchorRepository(path)


def test_selections_are_appended_with_their_rank_snapshot(repo):
    before = int(time.time() * 1000)
    assert repo.record(["a", "b", "", None], source="ladder_seed", region="EUW",
                       ranks={"a": {"leaguePoints": 1500, "hotStreak": True}}) == 2
    repo.record(["a"], source="ladder_seed", region="EUW", selected_at=before + 10_000)

    with sqlite3.connect(repo.db_path) as conn:
        rows = conn.execute("SELECT puuid, region, source, selected_at, rank_json FROM anchor_selections "
                            "ORDER BY id").fetchall()
    assert [(r[0], r[2]) for r in rows] == [("a", "ladder_seed"), ("b", "ladder_seed"), ("a", "ladder_seed")]
    assert before <= rows[0][3] <= rows[2][3]
    assert json.loads(rows[0][4]) == {"leaguePoints": 1500, "hotStreak": True}
    assert rows[1][4] is None


def test_unknown_sources_are_refused(repo):
    with pytest.raises(ValueError, match="source"):
        repo.record(["a"], source="vibes")


def test_only_selections_on_rank_count_and_the_latest_one_wins(repo):
    repo.record(["seed"], source="ladder_seed", selected_at=1000)
    repo.record(["seed"], source="top_leagues", selected_at=3000)
    repo.record(["seed"], source="manual_seed", selected_at=9000)
    repo.record(["fan"], source="prediction_warmup", selected_at=5000)
    repo.record(["old"], source="legacy_players_last_saved", selected_at=5000)

    assert repo.rank_selection_ms(["seed", "fan", "old", "absent", None]) == {"seed": 3000}
    assert {"ladder_seed", "top_leagues", "pro_import", "legacy_crawl_queue"} == RANK_SELECTION_SOURCES


def test_lookup_handles_more_puuids_than_sqlite_variables(repo):
    repo.record(["last"], source="pro_import", selected_at=7)
    assert repo.rank_selection_ms([f"p{i}" for i in range(2500)] + ["last"]) == {"last": 7}
