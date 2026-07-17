"""Deterministic French-to-AnalysisRequest planner with no SQL surface."""

from __future__ import annotations

import re
import unicodedata

from .cohorts import AnalyticsAbstention
from .registry import METRICS
from .types import AnalysisFilters, AnalysisRequest, Intent


REGION_PATTERNS = {
    "EUW": r"\beuw\b",
    "EUN": r"\beun\b",
    "NA": r"\bna\b|\bamerique du nord\b",
    "KR": r"\bkr\b|\bcoree\b",
}
TIMELINE_PATTERN = re.compile(r"\b(5|10|15)\s*minutes?\b|\bearly\b", re.I)
ESPORT_PATTERN = re.compile(
    r"\b(team|equipe|g2|fnatic|t1|gentle mates|karmine)\b", re.I
)
CAUSAL_PATTERN = re.compile(
    r"\b(cause|causal|garantit|prouve|certain|certaine|pari)\b", re.I
)
PATCH_PATTERN = re.compile(r"\b\d{1,2}\.\d{1,2}\b")
FILTER_FIELDS = {
    "regions",
    "patches",
    "queue_ids",
    "date_start_ms",
    "date_end_ms",
    "champion",
    "role",
}


def _normalized(message: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", message.lower())
        if not unicodedata.combining(char)
    )


def _contains_alias(text: str, alias: str) -> bool:
    value = re.escape(_normalized(alias))
    return re.search(rf"(?<!\w){value}(?!\w)", text) is not None


def _metric(text: str) -> str | None:
    aliases = sorted(
        (
            (_normalized(alias), spec.name)
            for spec in METRICS.values()
            for alias in spec.aliases
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    return next(
        (name for alias, name in aliases if _contains_alias(text, alias)), None
    )


def _validated_mapping(
    value: dict | None, allowed: set[str], label: str
) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"UNSUPPORTED_FILTER: {label}")
    return value


class FrenchPlanner:
    def plan(
        self,
        message: str,
        context: dict | None = None,
        filters: dict | None = None,
    ) -> AnalysisRequest:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("UNSUPPORTED_INTENT: empty message")
        context_value = _validated_mapping(
            context, {"intent", "metric", "filters"}, "context"
        )
        context_filters = _validated_mapping(
            context_value.get("filters"), FILTER_FIELDS, "context.filters"
        )
        if context_value:
            context_value = {**context_value, "filters": context_filters}
            try:
                previous = AnalysisRequest.from_dict(context_value)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("UNSUPPORTED_FILTER: context") from exc
        else:
            previous = None
        filter_value = _validated_mapping(filters, FILTER_FIELDS, "filters")

        normalized = _normalized(message)
        if TIMELINE_PATTERN.search(normalized):
            raise AnalyticsAbstention("TIMELINE_DATA_UNAVAILABLE")
        if ESPORT_PATTERN.search(normalized):
            raise AnalyticsAbstention("ESPORT_TEAM_IDENTITY_UNAVAILABLE")
        if CAUSAL_PATTERN.search(normalized):
            raise AnalyticsAbstention("UNSUPPORTED_INTENT")

        asks_evidence = "preuve" in normalized
        asks_limitations = "limite" in normalized
        if (asks_evidence or asks_limitations) and previous is None:
            raise AnalyticsAbstention("UNSUPPORTED_INTENT")

        regions = tuple(
            region
            for region, pattern in REGION_PATTERNS.items()
            if re.search(pattern, normalized, re.I)
        )
        patches = tuple(dict.fromkeys(PATCH_PATTERN.findall(normalized)))
        metric = _metric(normalized)
        if previous and normalized.startswith(("et ", "qu'en ", "et en ")):
            metric = metric or previous.metric
        if previous and (asks_evidence or asks_limitations):
            metric = previous.metric

        intent = Intent.EXPLAIN_METRIC if metric else Intent.RANK_CONDITIONS
        if asks_evidence:
            intent = Intent.SHOW_EVIDENCE
        elif asks_limitations:
            intent = Intent.SHOW_LIMITATIONS
        elif len(patches) == 2:
            intent = Intent.COMPARE_PATCHES
        elif len(regions) >= 2 or "compare" in normalized or "difference" in normalized:
            intent = Intent.COMPARE_REGIONS

        try:
            explicit = AnalysisFilters.from_dict(filter_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("UNSUPPORTED_FILTER: filters") from exc
        previous_filters = previous.filters if previous else AnalysisFilters()
        structured_regions = tuple(region.upper() for region in explicit.regions)
        structured_role = explicit.role.upper() if explicit.role else None
        resolved = AnalysisFilters(
            regions=regions or structured_regions or previous_filters.regions,
            patches=patches or explicit.patches or previous_filters.patches,
            queue_ids=explicit.queue_ids,
            date_start_ms=(
                explicit.date_start_ms
                if explicit.date_start_ms is not None
                else previous_filters.date_start_ms
            ),
            date_end_ms=(
                explicit.date_end_ms
                if explicit.date_end_ms is not None
                else previous_filters.date_end_ms
            ),
            champion=explicit.champion or previous_filters.champion,
            role=structured_role or previous_filters.role,
        )
        return AnalysisRequest(intent, metric, resolved)
