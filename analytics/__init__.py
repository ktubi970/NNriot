"""Explainable retrospective analytics for NNriot."""

from .config import AnalyticsConfig
from .registry import FeatureClass, MetricKind, MetricSpec, get_metric
from .types import (
    AbstentionCode,
    AnalysisFilters,
    AnalysisRequest,
    Estimate,
    EvidencePacket,
    Intent,
    ResolvedCohort,
)

__version__ = "1.0.0"

__all__ = [
    "AbstentionCode",
    "AnalysisFilters",
    "AnalysisRequest",
    "AnalyticsConfig",
    "Estimate",
    "EvidencePacket",
    "FeatureClass",
    "Intent",
    "MetricKind",
    "MetricSpec",
    "ResolvedCohort",
    "get_metric",
]
