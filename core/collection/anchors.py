"""AnchorRepository: who the collector anchored on, why and when (``anchor_selections``).

Every collection entry point records the players whose match lists it is about
to fetch. The table is append-only: a player selected twice has two rows, so the
first selection is never overwritten (``players.updated_at`` is only the last
profile refresh).

An anchor selected on rank was selected *because* it won recently: its matches
played before the selection over-represent its wins (54.3 % before entry vs
49.6 % after, audit of 25 Oct 2026). ``MatchIngestor`` keeps those matches out
of the database; selections from other sources are provenance only.
"""
from __future__ import annotations

import json
import time
from typing import Iterable, Mapping, Optional

from core import config, database

RANK_SELECTION_SOURCES = frozenset({
    "ladder_seed",          # run_crawler: top of the ladder, then deeper when the queue runs dry
    "top_leagues",          # collect_top_leagues: Challenger / Grandmaster / Master
    "pro_import",           # lolpros and Liquipedia accounts
    "legacy_crawl_queue",   # crawl_queue priority >= 5 before schema v17
})
OTHER_SOURCES = frozenset({
    "manual_seed",                # Riot IDs typed by a user or passed to the crawler
    "manual_batch",               # batch collection with smurfs
    "puuid_accounts",             # known accounts without a stated reason
    "prediction_warmup",          # stale players of a custom prediction
    "spectator",                  # participants of a live game
    "legacy_players_last_saved",  # players rows before v17, at their last save (not all anchors)
})
SOURCES = RANK_SELECTION_SOURCES | OTHER_SOURCES

# Stay below SQLite's historical 999 bound-parameter limit.
_LOOKUP_CHUNK = 500


class AnchorRepository:
    """Append-only registry of collection anchors."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = config.get_db_path(db_path, require_exists=False)

    def record(
        self,
        puuids: Iterable[Optional[str]],
        *,
        source: str,
        region: Optional[str] = None,
        selected_at: Optional[int] = None,
        ranks: Optional[Mapping[str, dict]] = None,
    ) -> int:
        """Record that *puuids* were selected now (or at *selected_at*, UTC ms)."""
        if source not in SOURCES:
            raise ValueError(f"Unknown anchor source: {source!r}")
        selected_at = int(time.time() * 1000) if selected_at is None else selected_at
        ranks = ranks or {}
        rows = [
            (p, region, source, selected_at,
             json.dumps(ranks[p], sort_keys=True) if ranks.get(p) else None)
            for p in dict.fromkeys(p.strip() for p in puuids if p and p.strip())
        ]
        if not rows:
            return 0
        with database.get_connection(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO anchor_selections (puuid, region, source, selected_at, rank_json) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def rank_selection_ms(self, puuids: Iterable[Optional[str]]) -> dict[str, int]:
        """Latest selection on rank (UTC ms) of each of *puuids* that has one."""
        wanted = sorted({p for p in puuids if p})
        selections: dict[str, int] = {}
        if not wanted:
            return selections
        sources = sorted(RANK_SELECTION_SOURCES)
        with database.get_connection(self.db_path) as conn:
            for start in range(0, len(wanted), _LOOKUP_CHUNK):
                chunk = wanted[start:start + _LOOKUP_CHUNK]
                selections.update(conn.execute(
                    f"""
                    SELECT puuid, MAX(selected_at) FROM anchor_selections
                    WHERE source IN ({",".join("?" for _ in sources)})
                      AND puuid IN ({",".join("?" for _ in chunk)})
                    GROUP BY puuid
                    """,
                    (*sources, *chunk),
                ).fetchall())
        return selections


def rank_snapshot(entry: Mapping) -> dict:
    """The ladder fields that explain a selection on rank, from a Riot league entry."""
    return {k: entry[k] for k in ("tier", "rank", "leaguePoints", "wins", "losses", "hotStreak") if k in entry}
