"""Typed public contracts for controlled analytics requests and evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


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
        data = data or {}
        return cls(
            regions=tuple(str(value) for value in data.get("regions", ())),
            patches=tuple(str(value) for value in data.get("patches", ())),
            queue_ids=tuple(int(value) for value in data.get("queue_ids", (420,))),
            date_start_ms=data.get("date_start_ms"),
            date_end_ms=data.get("date_end_ms"),
            champion=data.get("champion"),
            role=data.get("role"),
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
