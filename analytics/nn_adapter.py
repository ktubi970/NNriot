"""Narrow discovery-only boundary around the existing neural checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .registry import get_metric


@dataclass(frozen=True)
class DiscoveryCandidate:
    metric: str
    filters: dict
    discovery_match_ids: tuple[str, ...]
    discovery_only: bool
    checkpoint_independently_validated: bool


@dataclass(frozen=True)
class CandidateValidation:
    allowed: bool
    code: str | None
    evidence_id: str | None


class NNDiscoveryAdapter:
    """Expose candidates, never claims, from the current post-match NN.

    ``model_v2.keras`` is a post-match checkpoint tied to hashed training rows
    and trained on all rows currently marked for training. It therefore carries
    no independent performance or evidence claim in this analytics system.
    """

    def __init__(self, predict_candidates: Callable, checkpoint: str):
        self.predict_candidates = predict_candidates
        self.checkpoint = checkpoint

    def discover(self, rows: list[dict]) -> tuple[DiscoveryCandidate, ...]:
        if not rows:
            raise ValueError("SOURCE_DATA_INVALID")
        try:
            match_ids = tuple(sorted({str(row["match_id"]) for row in rows}))
        except (KeyError, TypeError) as exc:
            raise ValueError("SOURCE_DATA_INVALID") from exc
        candidates = []
        for raw in self.predict_candidates(rows):
            if not isinstance(raw, dict):
                raise ValueError("SOURCE_DATA_INVALID")
            metric = str(raw.get("metric", ""))
            get_metric(metric)
            filters = raw.get("filters", {})
            if not isinstance(filters, dict):
                raise ValueError("SOURCE_DATA_INVALID")
            candidates.append(
                DiscoveryCandidate(
                    metric=metric,
                    filters=dict(filters),
                    discovery_match_ids=match_ids,
                    discovery_only=True,
                    checkpoint_independently_validated=False,
                )
            )
        return tuple(candidates)


class CandidateValidationGate:
    """Require independent engine evidence before a candidate is usable."""

    def __init__(self, validate_with_engine: Callable):
        self.validate_with_engine = validate_with_engine

    def validate(
        self,
        candidate: DiscoveryCandidate,
        validation_match_ids: set[str],
    ) -> CandidateValidation:
        get_metric(candidate.metric)
        if not candidate.discovery_only:
            return CandidateValidation(
                False, "NN_DIRECT_EVIDENCE_FORBIDDEN", None
            )
        if set(candidate.discovery_match_ids) & validation_match_ids:
            return CandidateValidation(
                False, "DISCOVERY_VALIDATION_OVERLAP", None
            )
        if not validation_match_ids:
            return CandidateValidation(False, "INSUFFICIENT_MATCHES", None)
        result = self.validate_with_engine(
            {"metric": candidate.metric, "filters": dict(candidate.filters)},
            set(candidate.discovery_match_ids),
        )
        if not isinstance(result, dict):
            return CandidateValidation(False, "ENGINE_EVIDENCE_REQUIRED", None)
        evidence_id = result.get("evidence_id")
        allowed = bool(result.get("allowed")) and isinstance(evidence_id, str)
        if allowed and evidence_id:
            return CandidateValidation(True, None, evidence_id)
        code = (
            "ENGINE_EVIDENCE_REQUIRED"
            if result.get("allowed")
            else "EFFECT_NOT_ROBUST"
        )
        return CandidateValidation(False, code, None)
