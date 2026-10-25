"""Short-lived cache of result-end ordered history for live serving.

Composes the pinned ``core.pre_match_history`` replay instead of editing it.
Only decoded source records are cached: ``PlayerPriorsTracker`` lookups insert
defaultdict entries, so a replayed tracker is rebuilt for every request.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

from core.pre_match_history import build_target_history, decode_history_row, read_recent_rows
from core.serving_contract import ContractError

HISTORY_CACHE_TTL_SECONDS = 60.0
HISTORY_CACHE_MAX_ENTRIES = 128


class CachedHistoryProvider:
    """Reuse one player set's history window for live requests within a TTL.

    A cached window keeps its original ``as_of_ms``: priors stay strictly
    before the prediction, and provenance reports the window actually used.
    Requests with an explicit ``prediction_time_ms`` bypass the cache.
    """

    def __init__(self, inner, *, ttl_seconds: float = HISTORY_CACHE_TTL_SECONDS,
                 max_entries: int = HISTORY_CACHE_MAX_ENTRIES, clock=time.monotonic):
        self.inner = inner
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict = OrderedDict()

    @property
    def history_limit(self):
        return self.inner.history_limit

    @property
    def db_path(self):
        return self.inner.db_path

    def enrich(self, match: dict, *, include_timeline: bool) -> dict:
        if match.get('prediction_time_ms') is not None:
            return self.inner.enrich(match, include_timeline=include_timeline)
        identities = frozenset(p.get('puuid') for k in ('team_a', 'team_b')
                               for p in match.get(k, []) if p.get('puuid'))
        key = (identities, include_timeline)
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and now - entry[0] >= self.ttl_seconds:
                entry = None
        if entry is None:
            as_of = int(time.time() * 1000)
            entry = (now, as_of, self._load_records(identities, as_of, include_timeline))
            with self._lock:
                self._entries[key] = entry
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
        _, as_of, records = entry
        history = build_target_history(records, match, as_of_ms=as_of)
        result = history.enrich(match, include_timeline=include_timeline, as_of_ms=as_of)
        result['priors_provenance']['history_window_limit'] = self.history_limit
        return result

    def _load_records(self, identities, as_of: int, include_timeline: bool) -> list:
        # Same read as DatabaseHistoryProvider.enrich, kept outside the pinned module.
        records = []
        for row in read_recent_rows(self.inner.db_path, self.history_limit, before_ms=as_of,
                                    timelines=include_timeline, puuids=identities, history_only=True):
            try:
                records.append(decode_history_row(row))
            except ContractError:
                records.append(None)
        return records
