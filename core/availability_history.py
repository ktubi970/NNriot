"""Opt-in availability-aware history: only rows ingested strictly before the target starts.

The default replay (``core.pre_match_history``) assumes a Riot result was
observable at its result end. The 25 Oct 2026 audit found that assumption leaks:
a player's history reaches the database only once that player is crawled, often
after the target match. Schema v16 records ``matches.ingested_at``; these
providers drop every history row ingested at or after the target's start, and
every row whose ingestion time is unknown (NULL before v16).

This module composes ``core.pre_match_history`` and ``core.frozen_history``
without modifying them: both are bound by SHA-256 in model manifests and frozen
datasets. It is not wired to serving. Doing so means adding it to
``serving_contract.FEATURE_SOURCE_FILES`` and validating a model on features it
produced. On legacy data every row is NULL, so players start cold.
"""
from __future__ import annotations

import sqlite3
import time
from bisect import bisect_left
from contextlib import closing
from pathlib import Path
from typing import Iterable, Optional

from core.frozen_history import FrozenHistoryProvider
from core.pre_match_history import (DatabaseHistoryProvider, ResultOrderedHistory,
                                    build_target_history, decode_history_row, read_recent_rows)
from core.serving_contract import ContractError

AVAILABILITY_POLICY = 'ingested_strictly_before_start.v1'
AVAILABILITY_ASSUMPTION = 'Rows ingested strictly before start; rows with unknown ingestion time excluded'

# Stay below SQLite's historical 999 bound-parameter limit.
_LOOKUP_CHUNK = 500


def read_ingestion_times(db_path, match_ids: Iterable[str]) -> dict[str, Optional[int]]:
    """``matches.ingested_at`` per match, read-only; all None before schema v16."""
    wanted = sorted({m for m in match_ids if m})
    uri = Path(db_path).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute('PRAGMA query_only=ON')
        if 'ingested_at' not in {r[1] for r in conn.execute('PRAGMA table_info(matches)')}:
            return dict.fromkeys(wanted)
        times: dict[str, Optional[int]] = dict.fromkeys(wanted)
        for start in range(0, len(wanted), _LOOKUP_CHUNK):
            chunk = wanted[start:start + _LOOKUP_CHUNK]
            times.update(conn.execute(
                f'SELECT match_id, ingested_at FROM matches WHERE match_id IN ({",".join("?" for _ in chunk)})',
                chunk).fetchall())
    return times


def build_available_history(records, match, *, as_of_ms: int,
                            ingested_at: dict[str, Optional[int]]) -> tuple[ResultOrderedHistory, int]:
    """``build_target_history`` on the records ingested strictly before *as_of_ms*.

    Returns the history and the number of excluded records involving a target player.
    Undecodable records (None) pass through, as in the default replay.
    """
    identities = {p.get('puuid') for k in ('team_a', 'team_b') for p in match.get(k, []) if p.get('puuid')}
    kept, excluded = [], 0
    for rec in records:
        if rec is not None:
            ingested = ingested_at.get(rec['match_id'])
            if type(ingested) is not int or ingested >= as_of_ms:
                excluded += any(p.get('puuid') in identities for p in rec['participants'])
                continue
        kept.append(rec)
    return build_target_history(kept, match, as_of_ms=as_of_ms), excluded


def _enrich(history, excluded, match, *, include_timeline, as_of_ms, history_limit):
    enriched = history.enrich(match, include_timeline=include_timeline, as_of_ms=as_of_ms)
    enriched['priors_provenance'].update(
        availability_policy=AVAILABILITY_POLICY, availability_assumption=AVAILABILITY_ASSUMPTION,
        excluded_not_yet_ingested=excluded, history_window_limit=history_limit)
    return enriched


class IngestedDatabaseHistoryProvider(DatabaseHistoryProvider):
    """``DatabaseHistoryProvider`` restricted to rows ingested before the prediction time."""

    def enrich(self, match: dict, *, include_timeline: bool) -> dict:
        as_of = match.get('prediction_time_ms') or int(time.time() * 1000)
        if type(as_of) is not int:
            raise ContractError('prediction_time_ms must be integer UTC milliseconds')
        identities = {p.get('puuid') for k in ('team_a', 'team_b') for p in match.get(k, []) if p.get('puuid')}
        records = []
        for row in read_recent_rows(self.db_path, self.history_limit, before_ms=as_of, timelines=include_timeline,
                                    puuids=identities, history_only=True):
            try:
                records.append(decode_history_row(row))
            except ContractError:
                records.append(None)
        ingested = read_ingestion_times(self.db_path, (r['match_id'] for r in records if r is not None))
        history, excluded = build_available_history(records, match, as_of_ms=as_of, ingested_at=ingested)
        return _enrich(history, excluded, match, include_timeline=include_timeline, as_of_ms=as_of,
                       history_limit=self.history_limit)


class IngestedFrozenHistoryProvider(FrozenHistoryProvider):
    """``FrozenHistoryProvider`` restricted to rows ingested before each target's start.

    *ingested_at* maps match IDs to ``matches.ingested_at``, e.g. from
    ``read_ingestion_times`` on the source of the frozen rows.
    """

    def __init__(self, rows, *, ingested_at: dict[str, Optional[int]], history_limit=10000):
        super().__init__(rows, history_limit=history_limit)
        self.ingested_at = dict(ingested_at)

    def enrich(self, match, *, include_timeline):
        as_of = match.get('prediction_time_ms')
        if type(as_of) is not int or as_of <= 0:
            raise ContractError('Frozen replay requires positive prediction_time_ms')
        # Same creation window and candidate rows as FrozenHistoryProvider.enrich.
        stop = bisect_left(self.times, as_of)
        start = max(0, stop - self.history_limit)
        identities = {p.get('puuid') for k in ('team_a', 'team_b') for p in match.get(k, []) if p.get('puuid')}
        indexes = set()
        for puuid in identities:
            positions = self.positions.get(puuid, [])
            indexes.update(positions[bisect_left(positions, start):bisect_left(positions, stop)])
        indexes.update(self.invalid[bisect_left(self.invalid, start):bisect_left(self.invalid, stop)])
        facts = [self.facts[i] for i in sorted(indexes)]
        history, excluded = build_available_history(facts, match, as_of_ms=as_of, ingested_at=self.ingested_at)
        return _enrich(history, excluded, match, include_timeline=include_timeline, as_of_ms=as_of,
                       history_limit=self.history_limit)
