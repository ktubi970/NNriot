"""Validate filters and resolve statistically powered analytical cohorts."""

from __future__ import annotations

import re

from .config import AnalyticsConfig
from .db import connect_analytics
from .registry import get_metric
from .types import (
    MAX_CHAMPION_LENGTH,
    MAX_PATCHES,
    MAX_QUEUE_IDS,
    MAX_REGIONS,
    MAX_ROLE_LENGTH,
    AnalysisRequest,
    ResolvedCohort,
)


SUPPORTED_REGIONS = {"NA", "EUW", "EUN", "KR", "OTHER"}
SUPPORTED_ROLES = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY", "UNKNOWN"}
_PATCH_PATTERN = re.compile(r"^[0-9]+\.[0-9]+$")


class AnalyticsAbstention(RuntimeError):
    """Expected refusal to infer from an unsupported or underpowered cohort."""

    def __init__(self, code: str, details: dict | None = None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


class CohortResolver:
    def __init__(self, path: str, config: AnalyticsConfig | None = None):
        self.path = path
        self.config = config or AnalyticsConfig(analytics_path=path)

    def _validate(self, request: AnalysisRequest) -> None:
        if request.metric is not None:
            get_metric(request.metric)
        filters = request.filters
        if (
            len(filters.regions) > MAX_REGIONS
            or len(filters.patches) > MAX_PATCHES
            or len(filters.queue_ids) > MAX_QUEUE_IDS
            or len(set(filters.regions)) != len(filters.regions)
            or len(set(filters.patches)) != len(filters.patches)
            or len(set(filters.queue_ids)) != len(filters.queue_ids)
        ):
            raise ValueError("UNSUPPORTED_FILTER: unbounded values")
        if any(
            not isinstance(value, str) or len(value) > 16
            for value in (*filters.regions, *filters.patches)
        ):
            raise ValueError("UNSUPPORTED_FILTER: value length")
        if filters.champion is not None and (
            not isinstance(filters.champion, str)
            or not filters.champion
            or len(filters.champion) > MAX_CHAMPION_LENGTH
        ):
            raise ValueError("UNSUPPORTED_FILTER: champion")
        if filters.role is not None and (
            not isinstance(filters.role, str)
            or not filters.role
            or len(filters.role) > MAX_ROLE_LENGTH
        ):
            raise ValueError("UNSUPPORTED_FILTER: role")
        if set(filters.regions) - SUPPORTED_REGIONS:
            raise ValueError("UNSUPPORTED_FILTER: region")
        if filters.role and filters.role not in SUPPORTED_ROLES:
            raise ValueError("UNSUPPORTED_FILTER: role")
        if not filters.queue_ids or any(queue != 420 for queue in filters.queue_ids):
            raise ValueError("UNSUPPORTED_FILTER: queue")
        if any(not _PATCH_PATTERN.fullmatch(patch) for patch in filters.patches):
            raise ValueError("UNSUPPORTED_FILTER: patch")
        if (
            filters.date_start_ms is not None
            and filters.date_end_ms is not None
            and filters.date_start_ms > filters.date_end_ms
        ):
            raise ValueError("UNSUPPORTED_FILTER: date range")

    def where_clause(
        self,
        request: AnalysisRequest,
        patches: tuple[str, ...],
        alias: str = "m",
    ) -> tuple[str, list]:
        """Build a hard-coded SQL predicate and a separate parameter list."""

        self._validate(request)
        if alias != "m":
            raise ValueError("UNSUPPORTED_FILTER: SQL alias")
        conditions: list[str] = []
        params: list = []
        filters = request.filters

        conditions.append(
            f"{alias}.queue_id IN ({','.join('?' for _ in filters.queue_ids)})"
        )
        params.extend(filters.queue_ids)
        if patches:
            conditions.append(
                f"{alias}.patch IN ({','.join('?' for _ in patches)})"
            )
            params.extend(patches)
        if filters.regions:
            conditions.append(
                f"{alias}.region IN ({','.join('?' for _ in filters.regions)})"
            )
            params.extend(filters.regions)
        if filters.date_start_ms is not None:
            conditions.append(f"{alias}.game_creation_ms >= ?")
            params.append(int(filters.date_start_ms))
        if filters.date_end_ms is not None:
            conditions.append(f"{alias}.game_creation_ms <= ?")
            params.append(int(filters.date_end_ms))
        if filters.champion is not None:
            conditions.append("p.champion = ?")
            params.append(filters.champion)
        if filters.role is not None:
            conditions.append("p.role = ?")
            params.append(filters.role)
        return " WHERE " + " AND ".join(conditions), params

    def _counts(
        self, request: AnalysisRequest, patches: tuple[str, ...]
    ) -> tuple[int, int]:
        if not patches:
            return 0, 0
        where_sql, params = self.where_clause(request, patches)
        with connect_analytics(self.path, readonly=True) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT m.match_id), COUNT(DISTINCT p.puuid) "
                "FROM match_facts m "
                "JOIN player_facts p ON p.match_id=m.match_id"
                + where_sql,
                params,
            ).fetchone()
        return int(row[0]), int(row[1])

    def resolve(self, request: AnalysisRequest) -> ResolvedCohort:
        self._validate(request)
        patches = tuple(request.filters.patches)
        widened = False
        if not patches:
            with connect_analytics(self.path, readonly=True) as conn:
                available = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT patch FROM match_facts WHERE queue_id=420 "
                        "GROUP BY patch ORDER BY MAX(game_creation_ms) DESC"
                    )
                ]
            selected: list[str] = []
            for patch in available[:3]:
                selected.append(patch)
                matches, players = self._counts(request, tuple(selected))
                if (
                    matches >= self.config.min_matches
                    and players >= self.config.min_players
                ):
                    break
            patches = tuple(selected)
            widened = len(patches) > 1

        matches, players = self._counts(request, patches)
        if matches < self.config.min_matches:
            raise AnalyticsAbstention(
                "INSUFFICIENT_MATCHES",
                {"matches": matches, "required": self.config.min_matches},
            )
        if players < self.config.min_players:
            raise AnalyticsAbstention(
                "INSUFFICIENT_PLAYERS",
                {"players": players, "required": self.config.min_players},
            )
        filters = request.filters
        return ResolvedCohort(
            mode="CLASSIC",
            queue_ids=filters.queue_ids,
            patches=patches,
            regions=filters.regions,
            date_start_ms=filters.date_start_ms,
            date_end_ms=filters.date_end_ms,
            champion=filters.champion,
            role=filters.role,
            widened=widened,
            matches=matches,
            players=players,
        )
