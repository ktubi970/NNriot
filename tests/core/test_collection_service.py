import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from core import database
from core.collection.models import IngestionResult, CrawlItem
from core.collection.service import CollectionService
from core.collection.tracker import CollectionTracker


@pytest.fixture
def service(tmp_path):
    db_file = str(tmp_path / "test_service.db")
    database.init_db(db_file)
    tracker = CollectionTracker()
    return CollectionService(db_path=db_file, tracker=tracker)


def test_service_initialization(tmp_path):
    db_file = str(tmp_path / "init_test.db")
    custom_tracker = CollectionTracker()
    svc = CollectionService(db_path=db_file, tracker=custom_tracker)
    assert svc.db_path == db_file
    assert svc.tracker is custom_tracker
    assert svc.match_repo is not None
    assert svc.player_repo is not None
    assert svc.train_repo is not None
    assert svc.queue_repo is not None
    assert svc.ingestor is not None


def test_collect_player_matches_success(service):
    mock_api = MagicMock()
    mock_api.get_match_ids.return_value = ["EUW1_100"]

    service.api_factory = MagicMock(return_value=mock_api)
    service.ingestor = MagicMock()
    service.ingestor.ingest_match_ids.return_value = IngestionResult(
        matches_saved=1,
        records_saved=1,
        timelines_saved=1,
        embeddings_computed=1,
    )
    service.player_repo.increment_player_matches = MagicMock()

    res = service.collect_player_matches(puuid="puuid_test", region="EUW1", count=1)

    assert res.matches_saved == 1
    assert res.records_saved == 1
    mock_api.get_match_ids.assert_called_once_with(
        "puuid_test", count=1, start_time=None, queue=420, type="ranked"
    )
    service.player_repo.increment_player_matches.assert_called_once_with("puuid_test", 1)


def test_collect_player_matches_empty(service):
    mock_api = MagicMock()
    mock_api.get_match_ids.return_value = []
    service.api_factory = MagicMock(return_value=mock_api)
    service.player_repo.increment_player_matches = MagicMock()

    res = service.collect_player_matches(puuid="puuid_empty", region="KR", count=5)
    assert res.matches_saved == 0
    assert res.records_saved == 0
    service.player_repo.increment_player_matches.assert_not_called()


def test_collect_player_matches_exception(service):
    mock_api = MagicMock()
    mock_api.get_match_ids.side_effect = RuntimeError("Riot API network failure")
    service.api_factory = MagicMock(return_value=mock_api)

    res = service.collect_player_matches(puuid="puuid_err", region="NA", count=10)
    assert res.errors == 1
    assert res.matches_saved == 0


def test_collect_seed_players(service):
    mock_api = MagicMock()
    mock_api.get_puuid.side_effect = lambda name, tag: f"p_{name}" if name != "Unknown" else None
    service.api_factory = MagicMock(return_value=mock_api)

    with patch.object(service, "collect_player_matches") as mock_collect_matches:
        mock_collect_matches.return_value = IngestionResult(
            matches_saved=2, records_saved=2, timelines_saved=2
        )
        service.player_repo.save_player = MagicMock()

        seeds = [("Faker", "KR1"), ("Unknown", "KR1")]
        res = service.collect_seed_players(seeds, matches_per_player=5, days_back=7)

        assert res.matches_saved == 2
        assert res.records_saved == 2
        assert res.skipped == 1
        service.player_repo.save_player.assert_called_once_with("p_Faker", "Faker", "KR1")
        assert service.tracker.to_dict()["status"] == "completed"


def test_collect_by_puuid_accounts(service):
    accounts = [
        {"puuid": "p_pro1", "server": "EUW", "gamename": "Caps", "tagline": "EUW1"},
        {"puuid": "", "server": "NA", "gamename": "NoPuuid", "tagline": "NA1"},
    ]

    service.player_repo.save_player = MagicMock()
    with patch.object(service, "collect_player_matches") as mock_collect_matches:
        mock_collect_matches.return_value = IngestionResult(
            matches_saved=3, records_saved=3
        )

        stats = service.collect_by_puuid_accounts(accounts, matches_per_player=10)
        assert stats["processed"] == 1
        assert stats["saved"] == 3
        assert stats["skipped"] == 1
        assert stats["errors"] == 0
        service.player_repo.save_player.assert_called_once_with("p_pro1", "Caps", "EUW1")
        assert service.tracker.to_dict()["status"] == "completed"


def test_collect_top_leagues(service):
    mock_api = MagicMock()
    mock_api.get_top_players.return_value = [
        {"puuid": "p_challenger_1", "leaguePoints": 1200}
    ]
    mock_api.get_account_by_puuid.return_value = {
        "game_name": "Chovy",
        "tag_line": "KR1",
    }
    service.api_factory = MagicMock(return_value=mock_api)
    service.player_repo.save_player = MagicMock()

    with patch.object(service, "collect_player_matches") as mock_collect_matches:
        mock_collect_matches.return_value = IngestionResult(
            matches_saved=5, records_saved=5, timelines_saved=5
        )

        res = service.collect_top_leagues(
            regions=["KR"], total_players_per_region=1, matches_per_player=5, days_back=14
        )

        assert res.matches_saved == 5
        assert res.records_saved == 5
        # The ladder read time bounds what the anchor may contribute as targets.
        assert type(mock_collect_matches.call_args.kwargs["discovery_before_ms"]) is int
        service.player_repo.save_player.assert_called_once_with("p_challenger_1", "Chovy", "KR1")
        assert service.tracker.to_dict()["status"] == "completed"


def test_collect_batch_with_smurfs(service):
    mock_api = MagicMock()
    mock_api.get_puuid.return_value = "p_main"
    service.api_factory = MagicMock(return_value=mock_api)
    service.player_repo.save_player = MagicMock()

    with patch.object(service, "collect_player_matches") as mock_collect_matches:
        mock_collect_matches.return_value = IngestionResult(
            matches_saved=4, records_saved=4, timelines_saved=4
        )

        res = service.collect_batch_with_smurfs(
            player_list=[("MainPlayer", "EUW1")], sources=["liquipedia"], count=10
        )

        assert res.matches_saved == 4
        assert res.records_saved == 4
        service.player_repo.save_player.assert_called_once_with("p_main", "MainPlayer", "EUW1")
        assert service.tracker.to_dict()["status"] == "completed"


def test_run_crawler_seeds_and_snowballs(service):
    # Queue is initially empty
    assert service.queue_repo.get_stats()["pending"] == 0

    mock_api = MagicMock()
    mock_api.get_top_players.return_value = [{"puuid": "seed_puuid_1"}]
    service.api_factory = MagicMock(return_value=mock_api)

    # When crawling seed_puuid_1, returns 1 match saved and discovers new participant
    with patch.object(service, "collect_player_matches") as mock_collect_matches:
        mock_collect_matches.return_value = IngestionResult(
            matches_saved=1,
            records_saved=1,
            discovered_puuids=[("discovered_puuid_2", "EUW")],
        )

        res = service.run_crawler(
            target_matches=1,
            regions=["EUW"],
            batch_size=1,
        )

        assert res.matches_saved == 1
        # Seed was inserted and crawled
        # Snowballed discovered_puuid_2 was enqueued
        stats = service.queue_repo.get_stats()
        assert stats["processed"] >= 1
        assert stats["total"] >= 2  # seed + discovered participant
        assert service.tracker.to_dict()["status"] == "completed"


def test_run_crawler_respects_stop_event(service):
    mock_api = MagicMock()
    mock_api.get_top_players.return_value = []
    service.api_factory = MagicMock(return_value=mock_api)
    stop_event = threading.Event()
    stop_event.set()

    res = service.run_crawler(
        target_matches=100,
        regions=["EUW"],
        stop_event=stop_event,
    )

    assert res.matches_saved == 0
    assert service.tracker.to_dict()["status"] == "completed"


# --- Seeds and top-league anchors are discovery-only (v16) ----------------------------------

DAY_MS = 86_400_000


def _match(match_id, creation_ms, puuids):
    participants = [{
        "puuid": puuid, "teamId": 100 if i < 5 else 200,
        "teamPosition": "TOP" if i % 5 == 0 else "MIDDLE", "championName": "Ahri",
        "win": i < 5, "kills": 3, "deaths": 1, "assists": 2, "goldEarned": 5000,
    } for i, puuid in enumerate(puuids)]
    return {"metadata": {"matchId": match_id, "participants": list(puuids)}, "info": {
        "gameMode": "CLASSIC", "queueId": 420, "gameDuration": 1500, "gameCreation": creation_ms,
        "gameVersion": "16.18.1", "participants": participants,
        "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}]}}


def _riot(match_lists, details, top_players=()):
    api = MagicMock()
    api.get_top_players.return_value = [{"puuid": p} for p in top_players]
    api.get_account_by_puuid.return_value = {"game_name": "Chovy", "tag_line": "KR1"}
    api.get_match_ids.side_effect = lambda puuid, **_: list(match_lists.get(puuid, []))
    api.get_match_details_batch.side_effect = lambda ids, **_: {m: details[m] for m in ids if m in details}
    api.get_match_timelines_batch.return_value = {}
    return api


def _table(db, sql, *params):
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]


def test_collect_player_matches_passes_the_anchor_and_its_selection_time(service):
    mock_api = MagicMock()
    mock_api.get_match_ids.return_value = ["EUW1_100"]
    service.api_factory = MagicMock(return_value=mock_api)
    service.ingestor = MagicMock()
    service.ingestor.ingest_match_ids.return_value = IngestionResult()

    service.collect_player_matches(puuid="puuid_test", region="EUW1", count=1, discovery_before_ms=123)

    service.ingestor.ingest_match_ids.assert_called_once_with(
        ["EUW1_100"], api=mock_api, region="EUW",
        fetched_by_puuid="puuid_test", discovery_before_ms=123,
    )


def test_service_ingestor_reads_seeds_from_the_service_queue(service):
    assert service.ingestor.queue_repo is service.queue_repo


def test_run_crawler_keeps_seed_pre_entry_matches_out_of_targets_and_history(service):
    now = int(time.time() * 1000)
    snowball = [f"p{i}" for i in range(1, 10)]
    details = {
        # The seed's own recent list: played before the seed entered the queue.
        "EUW1_PRE": _match("EUW1_PRE", now - DAY_MS, ["seed", *snowball]),
        # The same streak, reached through a snowballed teammate's list.
        "EUW1_PRE_VIA_P1": _match("EUW1_PRE_VIA_P1", now - 2 * DAY_MS,
                                  ["seed", "p1", *[f"q{i}" for i in range(8)]]),
        # Played after the entry: an ordinary target.
        "EUW1_POST": _match("EUW1_POST", now + 3_600_000, ["seed", *snowball]),
    }
    service.api_factory = MagicMock(return_value=_riot(
        {"seed": ["EUW1_PRE"], "p1": ["EUW1_PRE", "EUW1_PRE_VIA_P1", "EUW1_POST"]},
        details, top_players=["seed"]))

    res = service.run_crawler(target_matches=1, regions=["EUW"], batch_size=5)

    assert res.matches_saved == 1
    assert res.discovery_only == 2
    stored = _table(service.db_path, "SELECT match_id, fetched_by_puuid, ingested_at FROM matches")
    assert [(r["match_id"], r["fetched_by_puuid"]) for r in stored] == [("EUW1_POST", "p1")]
    assert stored[0]["ingested_at"] >= now
    assert [r["match_id"] for r in _table(service.db_path, "SELECT match_id FROM training_dataset")] == ["EUW1_POST"]
    seed = _table(service.db_path, "SELECT priority, is_processed FROM crawl_queue WHERE puuid = 'seed'")
    assert seed == [{"priority": 10, "is_processed": 1}]
    # Participants of the discovery-only match still feed the frontier.
    queued = {r["puuid"] for r in _table(service.db_path, "SELECT puuid FROM crawl_queue WHERE priority = 0")}
    assert set(snowball) <= queued


def test_collect_top_leagues_only_discovers_the_frontier(service):
    now = int(time.time() * 1000)
    others = [f"p{i}" for i in range(1, 10)]
    details = {
        "KR_PRE": _match("KR_PRE", now - DAY_MS, ["anchor", *others]),
        "KR_PRE_LATER": _match("KR_PRE_LATER", now - 2 * DAY_MS,
                               ["anchor", "p1", *[f"q{i}" for i in range(8)]]),
    }
    service.api_factory = MagicMock(return_value=_riot(
        {"anchor": ["KR_PRE"], "p1": ["KR_PRE_LATER"]}, details, top_players=["anchor"]))

    res = service.collect_top_leagues(regions=["KR"], total_players_per_region=1,
                                      matches_per_player=5, days_back=14)

    assert (res.matches_saved, res.records_saved, res.discovery_only) == (0, 0, 1)
    assert _table(service.db_path, "SELECT match_id FROM matches") == []
    queue = {r["puuid"]: r for r in _table(service.db_path, "SELECT * FROM crawl_queue")}
    assert (queue["anchor"]["priority"], queue["anchor"]["is_processed"], queue["anchor"]["region"]) == (10, 1, "KR")
    assert {p for p, r in queue.items() if r["priority"] == 0 and r["is_processed"] == 0} == set(others)
    assert _table(service.db_path, "SELECT puuid, game_name FROM players") == [
        {"puuid": "anchor", "game_name": "Chovy"}]

    # The anchor is registered as a seed: its earlier matches stay out when a
    # snowballed player's list returns them later.
    later = service.collect_player_matches("p1", region="KR", count=5)
    assert (later.matches_saved, later.discovery_only) == (0, 1)
    assert _table(service.db_path, "SELECT match_id FROM matches") == []


def test_collect_top_leagues_keeps_an_existing_queue_entry_and_marks_it_crawled(service):
    service.queue_repo.enqueue_puuids(["anchor"], region="KR", priority=0)
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("UPDATE crawl_queue SET created_at = '2026-01-01 00:00:00' WHERE puuid = 'anchor'")
    now = int(time.time() * 1000)
    details = {"KR_PRE": _match("KR_PRE", now - DAY_MS, ["anchor", *[f"p{i}" for i in range(1, 10)]])}
    service.api_factory = MagicMock(return_value=_riot({"anchor": ["KR_PRE"]}, details, top_players=["anchor"]))

    res = service.collect_top_leagues(regions=["KR"], total_players_per_region=1, matches_per_player=5)

    # Its own fetched list is still discovery-only (selection time), and it is not re-crawled.
    assert (res.matches_saved, res.discovery_only) == (0, 1)
    row = _table(service.db_path, "SELECT priority, is_processed, created_at FROM crawl_queue WHERE puuid = 'anchor'")
    assert row == [{"priority": 0, "is_processed": 1, "created_at": "2026-01-01 00:00:00"}]
