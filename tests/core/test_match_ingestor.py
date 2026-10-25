import sqlite3
import time

import pytest
from unittest.mock import MagicMock, patch
from core import database
from core.collection.ingestor import MatchIngestor
from core.collection.models import IngestionResult, MatchQualityFilter


@pytest.fixture
def ingestor(tmp_path):
    db_file = str(tmp_path / "test_ingest.db")
    database.init_db(db_file)
    return MatchIngestor(db_path=db_file)


def _make_match_payload(game_mode="CLASSIC", queue_id=420, duration=600, winner_team=200):
    teams = [
        {"teamId": 100, "win": winner_team == 100},
        {"teamId": 200, "win": winner_team == 200},
    ]
    participants = []
    for i in range(10):
        t_id = 100 if i < 5 else 200
        participants.append({
            "puuid": f"puuid_{i}",
            "teamId": t_id,
            "teamPosition": "TOP" if i % 5 == 0 else "MIDDLE",
            "championName": "Ahri",
            "win": (t_id == winner_team),
            "kills": 3,
            "deaths": 1,
            "assists": 2,
            "goldEarned": 5000,
        })
    return {
        "info": {
            "gameMode": game_mode,
            "queueId": queue_id,
            "gameDuration": duration,
            "gameCreation": 1700000000000,
            "gameVersion": "14.1.1",
            "teams": teams,
            "participants": participants,
        }
    }


def test_build_training_record_leak_free(ingestor):
    details = _make_match_payload(winner_team=200)
    record = ingestor.build_training_record("EUW1_123", details, region="EUW")
    assert record is not None
    mid, feature, winner, labels = record
    assert mid == "EUW1_123"
    assert winner == 1
    # Verify zero post-game stats in features
    for p in feature["team_a"] + feature["team_b"]:
        assert "kda" not in p
        assert "gold" not in p
        assert "kills" not in p
        assert "deaths" not in p
        assert "assists" not in p
    assert feature["region"] == "EUW"
    assert isinstance(labels, dict)
    assert labels["winner"] == 1


def test_build_training_record_winner_team_100(ingestor):
    details = _make_match_payload(winner_team=100)
    record = ingestor.build_training_record("EUW1_100", details, region="EUW")
    assert record is not None
    mid, feature, winner, labels = record
    assert mid == "EUW1_100"
    assert winner == 0
    assert labels["winner"] == 0


def test_build_training_record_rejects_non_classic_and_short(ingestor):
    # Non-classic
    bad_mode = _make_match_payload(game_mode="ARAM")
    assert ingestor.build_training_record("M1", bad_mode) is None

    # Remake duration < 300
    remake = _make_match_payload(duration=180)
    assert ingestor.build_training_record("M2", remake) is None


def test_build_training_record_rejects_disallowed_queue(ingestor):
    # Flex queue (440) rejected by default filter
    flex_match = _make_match_payload(queue_id=440)
    assert ingestor.build_training_record("M3", flex_match) is None


def test_build_training_record_rejects_invalid_winner(ingestor):
    payload = _make_match_payload()
    # Both teams marked as losing
    payload["info"]["teams"][0]["win"] = False
    payload["info"]["teams"][1]["win"] = False
    assert ingestor.build_training_record("M4", payload) is None


def test_build_training_record_handles_malformed_payload(ingestor):
    assert ingestor.build_training_record("M5", {}) is None
    assert ingestor.build_training_record("M6", None) is None
    assert ingestor.build_training_record("M7", {"info": "invalid"}) is None


def test_custom_quality_filter(tmp_path):
    db_file = str(tmp_path / "custom_filter.db")
    database.init_db(db_file)
    custom_filter = MatchQualityFilter(
        allowed_game_modes=frozenset({"ARAM"}),
        allowed_queue_ids=frozenset({450}),
        min_duration_seconds=120,
    )
    custom_ingestor = MatchIngestor(db_path=db_file, quality_filter=custom_filter)

    aram_match = _make_match_payload(game_mode="ARAM", queue_id=450, duration=200)
    record = custom_ingestor.build_training_record("ARAM_1", aram_match)
    assert record is not None

    classic_match = _make_match_payload(game_mode="CLASSIC", queue_id=420, duration=600)
    assert custom_ingestor.build_training_record("CLASSIC_1", classic_match) is None


def test_ingest_match_ids_success(ingestor):
    mock_api = MagicMock()
    mock_api.get_match_details_batch.return_value = {
        "EUW1_1": _make_match_payload(),
        "EUW1_2": _make_match_payload(),
    }
    mock_api.get_match_timelines_batch.return_value = {
        "EUW1_1": {"info": {}},
        "EUW1_2": {"info": {}},
    }

    with patch("core.spatial_rnn.extract_and_store_timeline_embeddings") as mock_spatial:
        result = ingestor.ingest_match_ids(["EUW1_1", "EUW1_2"], api=mock_api, region="EUW")

    assert result.matches_saved == 2
    assert result.records_saved == 2
    assert result.timelines_saved == 2
    assert result.embeddings_computed == 2
    assert len(result.discovered_puuids) == 10
    assert result.discovered_puuids[0][1] == "EUW"
    assert mock_spatial.call_count == 2


def test_ingest_match_ids_skips_existing(ingestor):
    # Pre-save one match
    ingestor.match_repo.save_match("EUW1_EXISTS", _make_match_payload())

    mock_api = MagicMock()
    mock_api.get_match_details_batch.return_value = {
        "EUW1_NEW": _make_match_payload(),
    }
    mock_api.get_match_timelines_batch.return_value = {
        "EUW1_NEW": {"info": {}},
    }

    with patch("core.spatial_rnn.extract_and_store_timeline_embeddings"):
        result = ingestor.ingest_match_ids(["EUW1_EXISTS", "EUW1_NEW"], api=mock_api, region="EUW")

    # Only new match is fetched from API
    mock_api.get_match_details_batch.assert_called_once_with(["EUW1_NEW"], max_workers=5, verbose=False)
    assert result.matches_saved == 1
    assert result.records_saved == 1
    assert result.skipped == 1


def test_ingest_match_ids_quality_filtering(ingestor):
    mock_api = MagicMock()
    mock_api.get_match_details_batch.return_value = {
        "EUW1_VALID": _make_match_payload(duration=600),
        "EUW1_REMAKE": _make_match_payload(duration=150),
        "EUW1_ARAM": _make_match_payload(game_mode="ARAM"),
    }
    mock_api.get_match_timelines_batch.return_value = {
        "EUW1_VALID": {"info": {}},
    }

    with patch("core.spatial_rnn.extract_and_store_timeline_embeddings"):
        result = ingestor.ingest_match_ids(
            ["EUW1_VALID", "EUW1_REMAKE", "EUW1_ARAM"], api=mock_api, region="EUW"
        )

    assert result.matches_saved == 1
    assert result.records_saved == 1
    assert result.skipped == 2
    assert result.timelines_saved == 1


def test_ingest_match_ids_handles_empty_and_errors(ingestor):
    # Empty input list
    res_empty = ingestor.ingest_match_ids([], api=MagicMock(), region="EUW")
    assert res_empty.matches_saved == 0
    assert res_empty.skipped == 0
    assert res_empty.errors == 0

    # API exception
    mock_api = MagicMock()
    mock_api.get_match_details_batch.side_effect = RuntimeError("Riot API timeout")
    res_error = ingestor.ingest_match_ids(["M_FAIL"], api=mock_api, region="EUW")
    assert res_error.matches_saved == 0
    assert res_error.errors == 1


def test_build_training_record_rejects_missing_queue_id(ingestor):
    # Match with None queueId
    none_queue = _make_match_payload(queue_id=None)
    assert ingestor.build_training_record("M_NONE", none_queue) is None

    # Match without queueId key in info
    no_queue = _make_match_payload()
    del no_queue["info"]["queueId"]
    assert ingestor.build_training_record("M_MISSING", no_queue) is None


def test_ingest_match_ids_isolates_timeline_embedding_failure(ingestor):
    mock_api = MagicMock()
    mock_api.get_match_details_batch.return_value = {
        "EUW1_1": _make_match_payload(),
        "EUW1_2": _make_match_payload(),
    }
    mock_api.get_match_timelines_batch.return_value = {
        "EUW1_1": {"info": {}},
        "EUW1_2": {"info": {}},
    }

    # Simulate failure on first match, success on second match
    def mock_extract(match_id, timeline_data, game_creation_time, db_path=None):
        if match_id == "EUW1_1":
            raise RuntimeError("Spatial RNN model computation crashed")
        return 10

    with patch(
        "core.spatial_rnn.extract_and_store_timeline_embeddings", side_effect=mock_extract
    ):
        result = ingestor.ingest_match_ids(
            ["EUW1_1", "EUW1_2"], api=mock_api, region="EUW"
        )

    assert result.matches_saved == 2
    assert result.records_saved == 2
    assert result.timelines_saved == 2
    # Second match still computed successfully despite first match failing
    assert result.embeddings_computed == 1



# --- v16 provenance, anchors and sightings ------------------------------------------------------

from core.collection.anchors import AnchorRepository

CREATION_MS = 1700000000000  # 2023-11-14 22:13:20 UTC, the payload's gameCreation


def _api_for(details):
    api = MagicMock()
    api.get_match_details_batch.side_effect = lambda ids, **_: {m: details[m] for m in ids if m in details}
    api.get_match_timelines_batch.return_value = {}
    return api


def _stored(db, table, match_id):
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(f"SELECT * FROM {table} WHERE match_id = ?", (match_id,)).fetchone()
    return dict(row) if row else None


def _sightings(db):
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT match_id, fetched_by_puuid, disposition FROM match_sightings "
                            "ORDER BY rowid").fetchall()


def test_ingested_match_records_fetching_anchor_and_ingestion_time(ingestor):
    before = int(time.time() * 1000)
    result = ingestor.ingest_match_ids(["EUW1_1"], api=_api_for({"EUW1_1": _make_match_payload()}),
                                       region="EUW", fetched_by_puuid="puuid_3")
    after = int(time.time() * 1000)

    facts = _stored(ingestor.db_path, "match_facts", "EUW1_1")
    assert result.matches_saved == 1
    assert facts["fetched_by_puuid"] == "puuid_3" and facts["queue_id"] == 420
    assert before <= facts["ingested_at"] <= after
    training = _stored(ingestor.db_path, "training_dataset", "EUW1_1")
    assert training["feature_extractor"] == "json_utils.extract_pre_match_features"


@pytest.mark.parametrize("source, selected_at, discovery_only", [
    ("ladder_seed", CREATION_MS + 1000, True),          # seed selected after the match
    ("top_leagues", CREATION_MS + 1, True),
    ("pro_import", CREATION_MS + 86_400_000, True),
    ("legacy_crawl_queue", CREATION_MS + 1000, True),
    ("ladder_seed", CREATION_MS, False),                # selected at the match's creation
    ("manual_seed", CREATION_MS + 1000, False),         # not selected on rank
    ("legacy_players_last_saved", CREATION_MS + 1000, False),
])
def test_a_participant_selected_on_rank_after_the_match_makes_it_discovery_only(
        ingestor, source, selected_at, discovery_only):
    # The match comes from another player's list: the bias follows the anchor's participation.
    AnchorRepository(ingestor.db_path).record(["puuid_8"], source=source, region="EUW", selected_at=selected_at)

    result = ingestor.ingest_match_ids(["EUW1_1"], api=_api_for({"EUW1_1": _make_match_payload()}),
                                       region="EUW", fetched_by_puuid="puuid_1")

    assert result.discovery_only == int(discovery_only)
    assert (_stored(ingestor.db_path, "matches", "EUW1_1") is None) is discovery_only
    assert len(result.discovered_puuids) == 10
    assert _sightings(ingestor.db_path) == [
        ("EUW1_1", "puuid_1", "discovery_only" if discovery_only else "stored")]


@pytest.mark.parametrize("arrangement", ["original", "teams_swapped", "players_permuted"])
def test_discovery_only_decision_ignores_team_order_and_player_order(ingestor, arrangement):
    payload = _make_match_payload()
    parts = payload["info"]["participants"]
    if arrangement == "teams_swapped":
        for p in parts:
            p["teamId"] = 300 - p["teamId"]
        payload["info"]["teams"] = [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}]
    elif arrangement == "players_permuted":
        parts.reverse()
    AnchorRepository(ingestor.db_path).record(["puuid_2"], source="ladder_seed", selected_at=CREATION_MS + 1)

    result = ingestor.ingest_match_ids(["EUW1_1"], api=_api_for({"EUW1_1": payload}), region="EUW")

    assert result.discovery_only == 1
    assert result.matches_saved == 0


def test_discovery_only_matches_are_not_downloaded_again_by_a_later_process(ingestor):
    AnchorRepository(ingestor.db_path).record(["puuid_0"], source="ladder_seed", selected_at=CREATION_MS + 1)
    api = _api_for({"EUW1_OLD": _make_match_payload()})
    first = ingestor.ingest_match_ids(["EUW1_OLD"], api=api, region="EUW", fetched_by_puuid="puuid_0")
    later = MatchIngestor(db_path=ingestor.db_path)
    second = later.ingest_match_ids(["EUW1_OLD"], api=api, region="EUW", fetched_by_puuid="puuid_1")

    assert first.discovery_only == 1
    assert second.skipped == 1 and second.discovery_only == 0
    assert api.get_match_details_batch.call_count == 1
    assert _sightings(ingestor.db_path) == [("EUW1_OLD", "puuid_0", "discovery_only"),
                                            ("EUW1_OLD", "puuid_1", "discovery_only")]


def test_every_returned_match_is_sighted_with_what_was_done(ingestor):
    ingestor.match_repo.save_match("EUW1_KNOWN", _make_match_payload())
    api = _api_for({"EUW1_NEW": _make_match_payload(), "EUW1_ARAM": _make_match_payload(game_mode="ARAM")})

    ingestor.ingest_match_ids(["EUW1_KNOWN", "EUW1_NEW", "EUW1_ARAM", "EUW1_GONE"], api=api,
                              region="EUW", fetched_by_puuid="puuid_4")

    assert sorted(_sightings(ingestor.db_path)) == [
        ("EUW1_ARAM", "puuid_4", "rejected"), ("EUW1_GONE", "puuid_4", "missing"),
        ("EUW1_KNOWN", "puuid_4", "already_stored"), ("EUW1_NEW", "puuid_4", "stored")]


def test_a_failed_download_is_sighted_as_missing(ingestor):
    api = MagicMock()
    api.get_match_details_batch.side_effect = RuntimeError("Riot API timeout")

    result = ingestor.ingest_match_ids(["EUW1_1", "EUW1_2"], api=api, region="EUW")

    assert result.errors == 2
    assert _sightings(ingestor.db_path) == [("EUW1_1", None, "missing"), ("EUW1_2", None, "missing")]
