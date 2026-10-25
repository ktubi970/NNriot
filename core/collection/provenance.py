"""Collection provenance of a match: which participants the collector anchored on.

``CollectionService.run_crawler`` seeds ``crawl_queue`` with the top of each
ladder (priority 10, then 5 when the queue runs dry) and snowballs on every
participant (priority 0). Older collection paths (top leagues, pro lists,
manual batches) record their anchors in ``players`` instead. An anchor chosen
for its rank is chosen *because* it won recently, so the matches fetched from
its history before it was selected over-represent its wins.

These counts annotate matches for training-time neutralisation and for
balanced reporting only. They describe how a match reached the database and
must never become model inputs.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

# run_crawler enqueues get_top_players at 10, then at 5 when deepening; snowball is 0.
LADDER_SEED_MIN_PRIORITY = 5


def parse_sqlite_timestamp_ms(value: Optional[str | int]) -> Optional[int]:
    """UTC epoch milliseconds from a schema-v16 integer or a legacy CURRENT_TIMESTAMP string."""
    if type(value) is int:
        return value
    if not value:
        return None
    parsed = datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


@dataclass(frozen=True)
class CrawlRegistry:
    """Snapshot of who the collector anchored on.

    ``seed_entry_ms``: ladder seeds (crawl_queue priority >= 5) and when they entered the queue.
    ``anchor_saved_ms``: other ``players`` rows and their last save, i.e. the end of the
    window in which their own history was fetched.
    ``crawled``: crawl_queue rows already processed, seeds included.
    ``crawled_at_ms``: when each of them was processed, where recorded.
    """

    seed_entry_ms: Mapping[str, int]
    anchor_saved_ms: Mapping[str, int]
    crawled: frozenset[str]
    crawled_at_ms: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TeamProvenance:
    seeds: int = 0
    pre_entry_seeds: int = 0
    legacy_anchors: int = 0
    pre_entry_anchors: int = 0
    crawled: int = 0
    # Crawled at or after the match: their history reached the database after it was played.
    crawled_after: int = 0


@dataclass(frozen=True)
class MatchProvenance:
    team_a: TeamProvenance
    team_b: TeamProvenance

    @property
    def has_ladder_seed(self) -> bool:
        return self.team_a.seeds + self.team_b.seeds > 0

    @property
    def has_pre_entry_seed(self) -> bool:
        return self.team_a.pre_entry_seeds + self.team_b.pre_entry_seeds > 0

    @property
    def has_pre_entry_anchor(self) -> bool:
        """A participant was selected on rank after this match: its outcome is over-represented."""
        a, b = self.team_a, self.team_b
        return a.pre_entry_seeds + b.pre_entry_seeds + a.pre_entry_anchors + b.pre_entry_anchors > 0

    @property
    def crawled_imbalance(self) -> int:
        """Team B minus team A, the same orientation as ``winner == 1``."""
        return self.team_b.crawled - self.team_a.crawled

    @property
    def crawled_after_imbalance(self) -> int:
        return self.team_b.crawled_after - self.team_a.crawled_after

    @property
    def is_balanced(self) -> bool:
        """The reporting subset of the 24 Oct 2026 audit: no ladder seed, equal crawled counts."""
        return not self.has_ladder_seed and self.crawled_imbalance == 0

    def swapped(self) -> "MatchProvenance":
        return MatchProvenance(team_a=self.team_b, team_b=self.team_a)

    def as_row(self) -> dict[str, int]:
        row: dict[str, int] = {}
        for field in TeamProvenance.__dataclass_fields__:
            row[f"{field}_a"] = getattr(self.team_a, field)
            row[f"{field}_b"] = getattr(self.team_b, field)
        row["has_ladder_seed"] = int(self.has_ladder_seed)
        row["has_pre_entry_seed"] = int(self.has_pre_entry_seed)
        row["has_pre_entry_anchor"] = int(self.has_pre_entry_anchor)
        row["crawled_imbalance"] = self.crawled_imbalance
        row["crawled_after_imbalance"] = self.crawled_after_imbalance
        row["balanced"] = int(self.is_balanced)
        return row


def team_provenance(puuids: Iterable[Optional[str]], creation_ms: int, registry: CrawlRegistry) -> TeamProvenance:
    seeds = pre_seeds = anchors = pre_anchors = crawled = crawled_after = 0
    for puuid in puuids:
        if not puuid:
            continue
        entry = registry.seed_entry_ms.get(puuid)
        if entry is not None:
            seeds += 1
            pre_seeds += creation_ms < entry
        saved = registry.anchor_saved_ms.get(puuid)
        if saved is not None:
            anchors += 1
            pre_anchors += creation_ms < saved
        if puuid in registry.crawled:
            crawled += 1
            crawled_at = registry.crawled_at_ms.get(puuid)
            crawled_after += crawled_at is not None and crawled_at >= creation_ms
    return TeamProvenance(seeds, pre_seeds, anchors, pre_anchors, crawled, crawled_after)


def match_provenance(
    team_a: Iterable[Optional[str]],
    team_b: Iterable[Optional[str]],
    creation_ms: int,
    registry: CrawlRegistry,
) -> MatchProvenance:
    return MatchProvenance(
        team_a=team_provenance(team_a, creation_ms, registry),
        team_b=team_provenance(team_b, creation_ms, registry),
    )


def load_crawl_registry(db_path: str) -> CrawlRegistry:
    """Read crawl_queue and players through a read-only connection.

    Works on schema v16 (integer UTC ms, ``players.updated_at``) and on older
    databases (CURRENT_TIMESTAMP text, ``players.last_updated``).
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        player_columns = {r[1] for r in conn.execute("PRAGMA table_info(players)")}
        saved_column = "updated_at" if "updated_at" in player_columns else "last_updated"
        seeds: dict[str, int] = {}
        crawled: dict[str, Optional[int]] = {}
        for puuid, priority, processed, created_at, processed_at in conn.execute(
            "SELECT puuid, priority, is_processed, created_at, processed_at FROM crawl_queue WHERE puuid IS NOT NULL"
        ):
            if priority is not None and priority >= LADDER_SEED_MIN_PRIORITY:
                seeds[puuid] = parse_sqlite_timestamp_ms(created_at)
            if processed:
                crawled[puuid] = parse_sqlite_timestamp_ms(processed_at)
        anchors = {
            puuid: parse_sqlite_timestamp_ms(saved)
            for puuid, saved in conn.execute(f"SELECT puuid, {saved_column} FROM players")
            if puuid not in seeds and saved
        }
    return CrawlRegistry(
        seed_entry_ms=seeds,
        anchor_saved_ms=anchors,
        crawled=frozenset(crawled),
        crawled_at_ms={p: t for p, t in crawled.items() if t is not None},
    )
