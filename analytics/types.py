"""Typed public contracts for controlled analytics requests and evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


MAX_REGIONS = 4
MAX_PATCHES = 3
MAX_QUEUE_IDS = 1
MAX_CHAMPION_LENGTH = 64
MAX_ROLE_LENGTH = 16


def _bounded_sequence(
    data: dict[str, Any],
    key: str,
    default: tuple,
    expected_type: type,
    max_unique: int,
) -> tuple:
    raw = data.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) > max(max_unique, 2):
        raise ValueError(f"UNSUPPORTED_FILTER: {key}")
    if any(
        not isinstance(value, expected_type)
        or (expected_type is int and isinstance(value, bool))
        for value in raw
    ):
        raise ValueError(f"UNSUPPORTED_FILTER: {key}")
    values = tuple(dict.fromkeys(raw))
    if len(values) > max_unique:
        raise ValueError(f"UNSUPPORTED_FILTER: {key}")
    return values


def _optional_string(
    data: dict[str, Any], key: str, maximum: int
) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"UNSUPPORTED_FILTER: {key}")
    return value


class Intent(str, Enum):
    RANK_CONDITIONS = "rank_conditions"
    EXPLAIN_METRIC = "explain_metric"
    COMPARE_REGIONS = "compare_regions"
    COMPARE_PATCHES = "compare_patches"
    SHOW_EVIDENCE = "show_evidence"
    SHOW_LIMITATIONS = "show_limitations"


class AbstentionCode(str, Enum):
    UNSUPPORTED_INTENT = "UNSUPPORTED_INTENT"
    UNSUPPORTED_FILTER = "UNSUPPORTED_FILTER"
    TIMELINE_DATA_UNAVAILABLE = "TIMELINE_DATA_UNAVAILABLE"
    ESPORT_TEAM_IDENTITY_UNAVAILABLE = "ESPORT_TEAM_IDENTITY_UNAVAILABLE"
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"
    INSUFFICIENT_PLAYERS = "INSUFFICIENT_PLAYERS"
    SPARSE_COMPARISON_GROUP = "SPARSE_COMPARISON_GROUP"
    EFFECT_NOT_ROBUST = "EFFECT_NOT_ROBUST"
    ANALYTICS_DB_STALE = "ANALYTICS_DB_STALE"
    SOURCE_DATA_INVALID = "SOURCE_DATA_INVALID"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_OUTPUT_REJECTED = "LLM_OUTPUT_REJECTED"


@dataclass(frozen=True)
class AnalysisFilters:
    regions: tuple[str, ...] = ()
    patches: tuple[str, ...] = ()
    queue_ids: tuple[int, ...] = (420,)
    date_start_ms: int | None = None
    date_end_ms: int | None = None
    champion: str | None = None
    role: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnalysisFilters:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("UNSUPPORTED_FILTER: filters")
        regions = _bounded_sequence(
            data, "regions", (), str, MAX_REGIONS
        )
        patches = _bounded_sequence(
            data, "patches", (), str, MAX_PATCHES
        )
        queue_ids = _bounded_sequence(
            data, "queue_ids", (420,), int, MAX_QUEUE_IDS
        )
        for key in ("date_start_ms", "date_end_ms"):
            value = data.get(key)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"UNSUPPORTED_FILTER: {key}")
        return cls(
            regions=tuple(value.upper() for value in regions),
            patches=patches,
            queue_ids=queue_ids,
            date_start_ms=data.get("date_start_ms"),
            date_end_ms=data.get("date_end_ms"),
            champion=_optional_string(
                data, "champion", MAX_CHAMPION_LENGTH
            ),
            role=(
                role.upper()
                if (role := _optional_string(data, "role", MAX_ROLE_LENGTH))
                else None
            ),
        )


@dataclass(frozen=True)
class AnalysisRequest:
    intent: Intent
    metric: str | None = None
    filters: AnalysisFilters = field(default_factory=AnalysisFilters)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisRequest:
        return cls(
            intent=Intent(data["intent"]),
            metric=data.get("metric"),
            filters=AnalysisFilters.from_dict(data.get("filters")),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["intent"] = self.intent.value
        return result


@dataclass(frozen=True)
class ResolvedCohort:
    mode: str
    queue_ids: tuple[int, ...]
    patches: tuple[str, ...]
    regions: tuple[str, ...]
    date_start_ms: int | None
    date_end_ms: int | None
    champion: str | None
    role: str | None
    widened: bool
    matches: int
    players: int


@dataclass(frozen=True)
class Estimate:
    metric: str
    value: float | None
    unit: str
    baseline: float | None
    effect_size: float | None
    confidence_interval_95: tuple[float, float] | None
    p_value: float | None
    q_value: float | None = None
    events_by_group: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidencePacket:
    evidence_id: str
    question: str
    analysis_request: dict[str, Any]
    cohort: dict[str, Any]
    sample: dict[str, Any]
    estimate: dict[str, Any]
    method: dict[str, Any]
    robustness: dict[str, Any]
    recommendation: dict[str, Any]
    limitations: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
