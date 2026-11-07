import threading
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
