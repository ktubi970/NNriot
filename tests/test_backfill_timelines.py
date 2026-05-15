"""Test the timeline backfill script with mocked Riot API."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
import pytest

import database
import backfill_timelines as bt


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("RIOT_API_KEY", "test_key_123")


def _insert_match(db, match_id, game_mode="CLASSIC"):
    with database.get_connection(db) as conn:
        conn.execute(
            "INSERT INTO matches (match_id, game_mode) VALUES (?, ?)",
            (match_id, game_mode)
        )


def test_region_inference():
    assert bt._region_from_match_id("EUW1_123") == "EUW"
    assert bt._region_from_match_id("KR_456") == "KR"
    assert bt._region_from_match_id("NA1_789") == "NA"
    assert bt._region_from_match_id("UNKNOWN_111") is None
    assert bt._region_from_match_id("") is None
    assert bt._region_from_match_id(None) is None


def test_backfill_no_matches_returns_zero(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)

    stats = bt.backfill_timelines(db_path=db)
    assert stats["processed"] == 0
    assert stats["saved"] == 0


def test_backfill_fetches_and_saves_timeline(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)
    _insert_match(db, "EUW1_111")
    _insert_match(db, "EUW1_222")

    fake_timeline = {"metadata": {"matchId": "EUW1_111"}, "info": {"frames": []}}

    with patch.object(bt.riot_api.RiotAPI, "get_match_timelines_batch",
                      return_value={"EUW1_111": fake_timeline, "EUW1_222": fake_timeline}):
        stats = bt.backfill_timelines(db_path=db, chunk_size=10, sleep_between_chunks=0)

    assert stats["processed"] == 2
    assert stats["saved"] == 2
    assert stats["errors"] == 0

    # Verify timelines stored
    assert database.match_timeline_exists("EUW1_111", db_path=db) is True
    assert database.match_timeline_exists("EUW1_222", db_path=db) is True


def test_backfill_idempotent(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)
    _insert_match(db, "EUW1_111")

    fake_timeline = {"info": {"frames": []}}
    with patch.object(bt.riot_api.RiotAPI, "get_match_timelines_batch",
                      return_value={"EUW1_111": fake_timeline}):
        stats1 = bt.backfill_timelines(db_path=db, chunk_size=10, sleep_between_chunks=0)

    assert stats1["processed"] == 1

    # Second run should see no work
    with patch.object(bt.riot_api.RiotAPI, "get_match_timelines_batch") as mock_batch:
        stats2 = bt.backfill_timelines(db_path=db, chunk_size=10, sleep_between_chunks=0)
        mock_batch.assert_not_called()  # nothing to fetch

    assert stats2["processed"] == 0


def test_backfill_handles_unknown_region(tmp_path):
    """Match with an unrecognizable region prefix gets a NULL placeholder."""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    _insert_match(db, "BADREGION_999")

    stats = bt.backfill_timelines(db_path=db, chunk_size=10, sleep_between_chunks=0)
    assert stats["skipped_region_unknown"] == 1
    # NULL placeholder should be present so we don't loop forever
    assert database.match_timeline_exists("BADREGION_999", db_path=db) is True


def test_backfill_handles_riot_returning_none(tmp_path):
    """When the Riot API returns no timeline for a match, a NULL placeholder is stored."""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    _insert_match(db, "EUW1_GONE")

    # Riot returns empty dict — match_id not in the result
    with patch.object(bt.riot_api.RiotAPI, "get_match_timelines_batch", return_value={}):
        stats = bt.backfill_timelines(db_path=db, chunk_size=10, sleep_between_chunks=0)

    assert stats["processed"] == 1
    assert stats["errors"] == 1
    # NULL placeholder present
    assert database.match_timeline_exists("EUW1_GONE", db_path=db) is True


def test_backfill_respects_limit(tmp_path):
    """--limit stops after the requested count."""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    for i in range(10):
        _insert_match(db, f"EUW1_{i:03d}")

    fake_timeline = {"info": {"frames": []}}
    # Each call to the patched batch returns a full timeline for whatever IDs come in.
    def fake_batch(ids, **kwargs):
        return {mid: fake_timeline for mid in ids}

    with patch.object(bt.riot_api.RiotAPI, "get_match_timelines_batch", side_effect=fake_batch):
        stats = bt.backfill_timelines(db_path=db, chunk_size=3, limit=5, sleep_between_chunks=0)

    # At chunk_size=3 with limit=5, we should stop after the second chunk processes
    # which lands processed at 6 (>= 5). The early-termination guard takes effect there.
    assert stats["processed"] >= 5
    assert stats["processed"] <= 9  # well below 10
