import pytest
from dataclasses import FrozenInstanceError

from core.collection.regions import resolve_region, TAG_TO_REGION
from core.collection.models import CrawlItem, IngestionResult, MatchQualityFilter
import core.collection as collection_pkg


def test_resolve_region_known_tags():
    assert resolve_region("KR1") == "KR"
    assert resolve_region("kr") == "KR"
    assert resolve_region("KR2") == "KR"
    assert resolve_region("EUW1") == "EUW"
    assert resolve_region("euw") == "EUW"
    assert resolve_region("NA1") == "NA"
    assert resolve_region("na") == "NA"
    assert resolve_region("BR1") == "BR"
    assert resolve_region("EUN1") == "EUN"
    assert resolve_region("JP1") == "JP"
    assert resolve_region("OC1") == "OCE"
    assert resolve_region("OCE") == "OCE"
    assert resolve_region("LA1") == "LA1"
    assert resolve_region("LA2") == "LA2"
    assert resolve_region("RU") == "RU"
    assert resolve_region("TR1") == "TR"
    assert "BR" in TAG_TO_REGION


def test_resolve_region_fallbacks():
    assert resolve_region("UNKNOWN_TAG") == "EUW"
    assert resolve_region("3327") == "EUW"
    assert resolve_region("") == "EUW"
    assert resolve_region(None) == "EUW"


def test_crawl_item_dataclass_defaults():
    item = CrawlItem(region="EUW")
    assert item.region == "EUW"
    assert item.puuid is None
    assert item.game_name is None
    assert item.tag_line is None
    assert item.priority == 0
    assert item.is_processed == 0
    assert item.id is None


def test_crawl_item_dataclass_custom():
    item = CrawlItem(
        puuid="puuid-123",
        region="KR",
        game_name="Faker",
        tag_line="KR1",
        priority=5,
        is_processed=1,
        id=42,
    )
    assert item.puuid == "puuid-123"
    assert item.region == "KR"
    assert item.game_name == "Faker"
    assert item.tag_line == "KR1"
    assert item.priority == 5
    assert item.is_processed == 1
    assert item.id == 42


def test_ingestion_result_dataclass_defaults():
    res1 = IngestionResult()
    assert res1.matches_saved == 0
    assert res1.records_saved == 0
    assert res1.timelines_saved == 0
    assert res1.embeddings_computed == 0
    assert res1.skipped == 0
    assert res1.errors == 0
    assert res1.discovered_puuids == []

    # Verify default factory isolation
    res2 = IngestionResult()
    res1.discovered_puuids.append(("puuid-abc", "KR"))
    assert len(res2.discovered_puuids) == 0


def test_ingestion_result_dataclass_custom():
    res = IngestionResult(
        matches_saved=10,
        records_saved=10,
        timelines_saved=8,
        embeddings_computed=8,
        skipped=2,
        errors=1,
        discovered_puuids=[("p1", "EUW"), ("p2", "NA")],
    )
    assert res.matches_saved == 10
    assert res.records_saved == 10
    assert res.timelines_saved == 8
    assert res.embeddings_computed == 8
    assert res.skipped == 2
    assert res.errors == 1
    assert len(res.discovered_puuids) == 2


def test_match_quality_filter_defaults():
    qf = MatchQualityFilter()
    assert "CLASSIC" in qf.allowed_game_modes
    assert 420 in qf.allowed_queue_ids
    assert qf.min_duration_seconds == 300


def test_match_quality_filter_frozen():
    qf = MatchQualityFilter()
    with pytest.raises(FrozenInstanceError):
        qf.min_duration_seconds = 600


def test_collection_package_exports():
    expected_exports = {
        "resolve_region",
        "TAG_TO_REGION",
        "CrawlItem",
        "IngestionResult",
        "MatchQualityFilter",
        "CrawlQueueRepository",
        "CollectionTracker",
        "get_default_tracker",
        "MatchIngestor",
        "CollectionService",
    }
    for symbol in expected_exports:
        assert hasattr(collection_pkg, symbol), f"Missing export: {symbol}"
    assert set(collection_pkg.__all__) == expected_exports
