"""Unified collection system: region resolution, data models, crawl components, and trackers."""

from core.collection.regions import resolve_region, TAG_TO_REGION
from core.collection.models import CrawlItem, IngestionResult, MatchQualityFilter
from core.collection.queue import CrawlQueueRepository
from core.collection.tracker import CollectionTracker, get_default_tracker
from core.collection.ingestor import MatchIngestor

__all__ = [
    "resolve_region",
    "TAG_TO_REGION",
    "CrawlItem",
    "IngestionResult",
    "MatchQualityFilter",
    "CrawlQueueRepository",
    "CollectionTracker",
    "get_default_tracker",
    "MatchIngestor",
]
