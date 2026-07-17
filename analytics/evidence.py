"""Canonical evidence packets and the single recommendation policy."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Iterable

from .config import AnalyticsConfig
from .statistics import StatisticalResult
from .types import AnalysisRequest, EvidencePacket, ResolvedCohort


def classify_recommendation(
    *,
    matches: int,
    players: int,
    group_counts: tuple[int, ...],
    ci: tuple[float, float],
    q_value: float,
    effect_size: float | None,
    win_rate_difference: float | None,
    temporal_directions: tuple[int, ...],
) -> dict:
    """Apply the approved statistical, practical, and stability gates once."""

    excludes_null = ci[0] > 0 or ci[1] < 0
    practical = (
        win_rate_difference is not None and abs(win_rate_difference) >= 0.03
    ) or (effect_size is not None and abs(effect_size) >= 0.20)
    nonzero = [direction for direction in temporal_directions if direction != 0]
    stable_two = len(nonzero) >= 2 and max(
        nonzero.count(1), nonzero.count(-1)
    ) >= 2
    groups_powered = bool(group_counts) and min(group_counts) >= 100
    moderate = (
        matches >= 500
        and players >= 100
        and groups_powered
        and excludes_null
        and q_value <= 0.05
        and practical
        and stable_two
    )
    if not moderate:
        return {
            "allowed": False,
            "strength": "none",
            "wording_key": "descriptive_only",
        }

    direction = 1 if sum(nonzero) > 0 else -1
    strong_practical = (
        win_rate_difference is not None and abs(win_rate_difference) >= 0.05
    ) or (effect_size is not None and abs(effect_size) >= 0.50)
    strong = (
        matches >= 2000
        and q_value <= 0.01
        and strong_practical
        and len(nonzero) == len(temporal_directions)
        and all(value == direction for value in nonzero)
    )
    return {
        "allowed": True,
        "strength": "strong" if strong else "moderate",
        "wording_key": (
            "association_strong" if strong else "association_moderate"
        ),
    }


def stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> object:
    """Normalize tuples and enums through the strict JSON representation."""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def evidence_request_hash(request: AnalysisRequest, question: str) -> str:
    return stable_hash(
        {
            "analysis_request": request.to_dict(),
            "question": question.strip(),
        }
    )


def packet_from_dict(value: dict) -> EvidencePacket:
    data = dict(value)
    data["limitations"] = tuple(data.get("limitations", ()))
    return EvidencePacket(**data)


def build_evidence(
    *,
    question: str,
    request: AnalysisRequest,
    cohort: ResolvedCohort,
    metric_name: str,
    unit: str,
    result: StatisticalResult,
    q_value: float,
    temporal_folds: tuple[dict, ...],
    match_ids: Iterable[str],
    dataset_version: str,
    method_name: str,
    family_size: int,
    config: AnalyticsConfig,
    group_counts: tuple[int, ...],
    win_rate_difference: float | None = None,
    additional_limitations: tuple[str, ...] = (),
) -> EvidencePacket:
    """Build a complete immutable evidence contract from numerical results."""

    temporal_directions = tuple(int(fold["direction"]) for fold in temporal_folds)
    match_id_list = list(match_ids)
    matches = len(match_id_list)
    recommendation = classify_recommendation(
        matches=matches,
        players=cohort.players,
        group_counts=group_counts,
        ci=result.confidence_interval_95,
        q_value=q_value,
        effect_size=result.effect_size,
        win_rate_difference=win_rate_difference,
        temporal_directions=temporal_directions,
    )

    limitations = ["RETROSPECTIVE_FINAL_STATE_ASSOCIATION_ONLY"]
    if cohort.widened:
        limitations.append("COHORT_WIDENED_TO_OLDER_PATCHES")
    if result.effect_size is None:
        limitations.append("EFFECT_SIZE_UNAVAILABLE_ZERO_VARIANCE")
    if not recommendation["allowed"]:
        limitations.append("EFFECT_NOT_ROBUST")
    limitations.extend(additional_limitations)
    limitations = list(dict.fromkeys(limitations))

    analysis_request = _canonical(request.to_dict())
    cohort_payload = {
        "mode": cohort.mode,
        "queue_ids": list(cohort.queue_ids),
        "patches": list(cohort.patches),
        "regions": list(cohort.regions),
        "date_range": {
            "start_ms": cohort.date_start_ms,
            "end_ms": cohort.date_end_ms,
        },
        "champion": cohort.champion,
        "role": cohort.role,
        "widened": cohort.widened,
    }
    estimate = {
        "metric": metric_name,
        "value": result.value,
        "unit": unit,
        "baseline": result.baseline,
        "effect_size": result.effect_size,
        "confidence_interval_95": list(result.confidence_interval_95),
        "p_value": result.p_value,
        "q_value": float(q_value),
        "odds_ratio": result.odds_ratio,
    }
    method = {
        "name": method_name,
        "version": config.method_version,
        "seed": config.seed,
        "bootstrap_resamples": config.bootstrap_resamples,
        "multiple_testing": "benjamini_hochberg",
        "candidate_family_size": family_size,
    }
    request_hash = evidence_request_hash(request, question)
    evidence_id = stable_hash(
        {
            "dataset_version": dataset_version,
            "request_hash": request_hash,
            "method_version": config.method_version,
            "seed": config.seed,
            "estimate": estimate,
        }
    )
    packet = EvidencePacket(
        evidence_id=evidence_id,
        question=question.strip(),
        analysis_request=analysis_request,
        cohort=cohort_payload,
        sample={
            "matches": matches,
            "players": cohort.players,
            "events_by_group": dict(result.events_by_group),
        },
        estimate=estimate,
        method=method,
        robustness={
            "temporal_folds": [dict(fold) for fold in temporal_folds],
            "regional_effects": [],
            "player_disjoint_check": {"available": False},
            "passed": recommendation["allowed"],
        },
        recommendation=recommendation,
        limitations=tuple(limitations),
        provenance={
            "dataset_version": dataset_version,
            "source_match_ids_hash": stable_hash(sorted(match_id_list)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    # Fail here, before caching, if any future field introduces NaN/Infinity.
    _canonical(packet.to_dict())
    return packet
