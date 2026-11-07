from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrawlItem:
    region: str
    puuid: Optional[str] = None
    game_name: Optional[str] = None
    tag_line: Optional[str] = None
    priority: int = 0
    is_processed: int = 0
    id: Optional[int] = None


@dataclass
class IngestionResult:
    matches_saved: int = 0
    records_saved: int = 0
    timelines_saved: int = 0
    embeddings_computed: int = 0
    skipped: int = 0
    errors: int = 0
    discovered_puuids: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class MatchQualityFilter:
    allowed_game_modes: frozenset[str] = frozenset({"CLASSIC"})
    allowed_queue_ids: frozenset[int] = frozenset({420})
    min_duration_seconds: int = 300
