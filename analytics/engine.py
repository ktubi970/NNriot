"""Execute allowlisted analyses and cache complete evidence packets."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .cohorts import AnalyticsAbstention, CohortResolver
from .config import AnalyticsConfig
from .db import EvidenceRepository, connect_analytics, get_meta
from .evidence import (
    build_evidence,
    evidence_request_hash,
    packet_from_dict,
)
from .registry import METRICS, MetricKind, MetricSpec, get_metric
from .regions import RegionalObservation, compare_regions
from .statistics import (
    StatisticalResult,
    benjamini_hochberg,
    binary_effect,
    continuous_effect,
)
from .types import AnalysisRequest, EvidencePacket, Intent, ResolvedCohort


FAMILIES = {
    "rank_conditions": tuple(METRICS),
    "explain_metric": None,
    "compare_regions": None,
    "compare_patches": None,
}


@dataclass(frozen=True)
class _MetricRun:
    spec: MetricSpec
    result: StatisticalResult
    match_ids: list[str]
    times: np.ndarray
    primary: np.ndarray
    outcome: np.ndarray | None
    temporal_folds: tuple[dict, ...]
    group_counts: tuple[int, ...]
    method_name: str


def _direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _binary_rate(exposure: np.ndarray, outcome: np.ndarray) -> float:
    exposed = outcome[exposure == 1]
    comparison = outcome[exposure == 0]
    if len(exposed) == 0 or len(comparison) == 0:
        return float("nan")
    return float(exposed.mean() - comparison.mean())


def _temporal_folds(
    primary: np.ndarray, outcome: np.ndarray | None
) -> tuple[dict, ...]:
    folds: list[dict] = []
    for fold_number, indices in enumerate(
        np.array_split(np.arange(len(primary)), 3), start=1
    ):
        if outcome is None:
            value = float(primary[indices].mean())
        else:
            value = _binary_rate(primary[indices], outcome[indices])
        folds.append(
            {
                "fold": fold_number,
                "matches": int(len(indices)),
                "value": None if not math.isfinite(value) else value,
                "direction": 0 if not math.isfinite(value) else _direction(value),
            }
        )
    return tuple(folds)


class AnalysisEngine:
    def __init__(self, path: str, config: AnalyticsConfig | None = None):
        self.path = path
        self.config = config or AnalyticsConfig(analytics_path=path)
        self.resolver = CohortResolver(path, self.config)
        self.repository = EvidenceRepository(path)

    def _dataset_version(self) -> str:
        version = get_meta(self.path, "dataset_version")
        if not version:
            raise AnalyticsAbstention("ANALYTICS_DB_STALE")
        return version

    def _query_metric(
        self,
        request: AnalysisRequest,
        cohort: ResolvedCohort,
        spec: MetricSpec,
    ) -> _MetricRun:
        where_sql, params = self.resolver.where_clause(request, cohort.patches)
        needs_players = (
            request.filters.champion is not None
            or request.filters.role is not None
        )
        player_join = (
            " JOIN player_facts p ON p.match_id=m.match_id" if needs_players else ""
        )
        column = spec.column
        with connect_analytics(self.path, readonly=True) as conn:
            if spec.kind is MetricKind.CONTINUOUS:
                rows = conn.execute(
                    "SELECT DISTINCT m.match_id,m.game_creation_ms,"
                    f"winner.{column} AS winner_value,"
                    f"loser.{column} AS loser_value "
                    "FROM match_facts m "
                    "JOIN team_facts winner ON winner.match_id=m.match_id "
                    "AND winner.win=1 "
                    "JOIN team_facts loser ON loser.match_id=m.match_id "
                    "AND loser.win=0"
                    + player_join
                    + where_sql
                    + " ORDER BY m.game_creation_ms,m.match_id",
                    params,
                ).fetchall()
                match_ids = [str(row["match_id"]) for row in rows]
                times = np.asarray(
                    [int(row["game_creation_ms"]) for row in rows], dtype=np.int64
                )
                primary = np.asarray(
                    [
                        float(row["winner_value"]) - float(row["loser_value"])
                        for row in rows
                    ],
                    dtype=float,
                )
                try:
                    result = continuous_effect(
                        primary,
                        seed=self.config.seed,
                        resamples=self.config.bootstrap_resamples,
                    )
                except ValueError as exc:
                    raise AnalyticsAbstention(str(exc)) from exc
                return _MetricRun(
                    spec=spec,
                    result=result,
                    match_ids=match_ids,
                    times=times,
                    primary=primary,
                    outcome=None,
                    temporal_folds=_temporal_folds(primary, None),
                    group_counts=(len(primary), len(primary)),
                    method_name="paired_winner_loser_bootstrap",
                )

            rows = conn.execute(
                "SELECT DISTINCT m.match_id,m.game_creation_ms,"
                f"blue.{column} AS exposure,blue.win AS outcome "
                "FROM match_facts m "
                "JOIN team_facts blue ON blue.match_id=m.match_id "
                "AND blue.team_id=100"
                + player_join
                + where_sql
                + " ORDER BY m.game_creation_ms,m.match_id",
                params,
            ).fetchall()
        match_ids = [str(row["match_id"]) for row in rows]
        times = np.asarray(
            [int(row["game_creation_ms"]) for row in rows], dtype=np.int64
        )
        primary = np.asarray([int(row["exposure"]) for row in rows], dtype=np.int8)
        outcome = np.asarray([int(row["outcome"]) for row in rows], dtype=np.int8)
        try:
            result = binary_effect(
                primary,
                outcome,
                min_group=self.config.min_group_events,
            )
        except ValueError as exc:
            code = str(exc)
            if "SPARSE_COMPARISON_GROUP" in code:
                raise AnalyticsAbstention(
                    "SPARSE_COMPARISON_GROUP",
                    {
                        "exposed": int((primary == 1).sum()),
                        "comparison": int((primary == 0).sum()),
                    },
                ) from exc
            raise AnalyticsAbstention("SOURCE_DATA_INVALID") from exc
        return _MetricRun(
            spec=spec,
            result=result,
            match_ids=match_ids,
            times=times,
            primary=primary,
            outcome=outcome,
            temporal_folds=_temporal_folds(primary, outcome),
            group_counts=(
                result.events_by_group["exposed"],
                result.events_by_group["comparison"],
            ),
            method_name="binary_blue_exposure_exact",
        )

    def _cached(
        self,
        dataset_version: str,
        request: AnalysisRequest,
        question: str,
    ) -> EvidencePacket | None:
        value = self.repository.get(
            dataset_version,
            evidence_request_hash(request, question),
            self.config.method_version,
        )
        if not value:
            return None
        packet = packet_from_dict(value)
        if (
            packet.method.get("seed") != self.config.seed
            or packet.method.get("bootstrap_resamples")
            != self.config.bootstrap_resamples
        ):
            return None
        return packet

    def _store(
        self,
        packet: EvidencePacket,
        dataset_version: str,
        request: AnalysisRequest,
        question: str,
    ) -> None:
        self.repository.put(
            packet.evidence_id,
            dataset_version,
            evidence_request_hash(request, question),
            self.config.method_version,
            packet.to_dict(),
        )

    def _packet(
        self,
        run: _MetricRun,
        request: AnalysisRequest,
        cohort: ResolvedCohort,
        question: str,
        dataset_version: str,
        q_value: float,
        family_size: int,
        read_cache: bool = True,
    ) -> EvidencePacket:
        if read_cache:
            cached = self._cached(dataset_version, request, question)
            if cached is not None:
                return cached
        limitations: list[str] = []
        if run.spec.kind is MetricKind.BINARY and run.result.odds_ratio is None:
            limitations.append("ODDS_RATIO_UNAVAILABLE_ZERO_CELL")
        packet = build_evidence(
            question=question,
            request=request,
            cohort=cohort,
            metric_name=run.spec.name,
            unit=run.spec.unit,
            result=run.result,
            q_value=q_value,
            temporal_folds=run.temporal_folds,
            match_ids=run.match_ids,
            dataset_version=dataset_version,
            method_name=run.method_name,
            family_size=family_size,
            config=self.config,
            group_counts=run.group_counts,
            win_rate_difference=(
                run.result.value if run.spec.kind is MetricKind.BINARY else None
            ),
            additional_limitations=tuple(limitations),
        )
        self._store(packet, dataset_version, request, question)
        return packet

    def _standard_run(
        self,
        request: AnalysisRequest,
        question: str,
        cohort: ResolvedCohort | None,
        read_cache: bool = True,
    ) -> list[EvidencePacket]:
        if request.intent is Intent.RANK_CONDITIONS:
            metric_names = tuple(METRICS)
        elif request.intent is Intent.EXPLAIN_METRIC and request.metric:
            metric_names = (request.metric,)
        else:
            raise AnalyticsAbstention("UNSUPPORTED_INTENT")

        resolved = cohort or self.resolver.resolve(request)
        dataset_version = self._dataset_version()
        if read_cache and len(metric_names) == 1:
            cached = self._cached(dataset_version, request, question)
            if cached is not None:
                return [cached]

        runs: list[_MetricRun] = []
        candidate_requests: list[AnalysisRequest] = []
        for metric_name in metric_names:
            spec = get_metric(metric_name)
            candidate_request = (
                request
                if len(metric_names) == 1
                else AnalysisRequest(request.intent, metric_name, request.filters)
            )
            runs.append(self._query_metric(candidate_request, resolved, spec))
            candidate_requests.append(candidate_request)

        q_values = benjamini_hochberg([run.result.p_value for run in runs])
        return [
            self._packet(
                run,
                candidate_request,
                resolved,
                question,
                dataset_version,
                q_value,
                len(runs),
                read_cache=read_cache,
            )
            for run, candidate_request, q_value in zip(
                runs, candidate_requests, q_values
            )
        ]

    def _patch_request(self, request: AnalysisRequest, patch: str) -> AnalysisRequest:
        filters = replace(request.filters, patches=(patch,))
        return AnalysisRequest(Intent.EXPLAIN_METRIC, request.metric, filters)

    def _run_estimate(self, run: _MetricRun, indices: np.ndarray) -> float:
        if run.outcome is None:
            return float(run.primary[indices].mean())
        return _binary_rate(run.primary[indices], run.outcome[indices])

    def _patch_contrast_result(
        self,
        first: _MetricRun,
        second: _MetricRun,
        first_patch: str,
        second_patch: str,
    ) -> tuple[StatisticalResult, tuple[dict, ...]]:
        rng = np.random.default_rng(self.config.seed)
        count = self.config.bootstrap_resamples

        def estimate(run: _MetricRun, indices: np.ndarray) -> float:
            return self._run_estimate(run, indices)

        first_all = np.arange(len(first.primary))
        second_all = np.arange(len(second.primary))
        first_effect = estimate(first, first_all)
        second_effect = estimate(second, second_all)
        observed = second_effect - first_effect
        bootstrap = np.empty(count, dtype=float)
        for index in range(count):
            first_draw = rng.integers(0, len(first.primary), len(first.primary))
            second_draw = rng.integers(0, len(second.primary), len(second.primary))
            bootstrap[index] = estimate(second, second_draw) - estimate(
                first, first_draw
            )

        if first.outcome is None:
            combined_primary = np.concatenate([first.primary, second.primary])
            combined_outcome = None
        else:
            combined_primary = np.concatenate([first.primary, second.primary])
            combined_outcome = np.concatenate([first.outcome, second.outcome])
        first_size = len(first.primary)
        permuted = np.empty(count, dtype=float)
        for index in range(count):
            order = rng.permutation(len(combined_primary))
            left, right = order[:first_size], order[first_size:]
            if combined_outcome is None:
                left_effect = float(combined_primary[left].mean())
                right_effect = float(combined_primary[right].mean())
            else:
                left_effect = _binary_rate(
                    combined_primary[left], combined_outcome[left]
                )
                right_effect = _binary_rate(
                    combined_primary[right], combined_outcome[right]
                )
            permuted[index] = abs(right_effect - left_effect)
        finite_permuted = permuted[np.isfinite(permuted)]
        p_value = (1.0 + float(np.count_nonzero(finite_permuted >= abs(observed)))) / (
            len(finite_permuted) + 1.0
        )
        finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
        if len(finite_bootstrap) < count * 0.95:
            raise AnalyticsAbstention("SPARSE_COMPARISON_GROUP")
        ci_values = np.quantile(finite_bootstrap, [0.025, 0.975])

        if first.outcome is None:
            pooled_numerator = (
                (len(first.primary) - 1) * first.primary.var(ddof=1)
                + (len(second.primary) - 1) * second.primary.var(ddof=1)
            )
            pooled_denominator = len(first.primary) + len(second.primary) - 2
            pooled = math.sqrt(float(pooled_numerator / pooled_denominator))
            effect_size = observed / pooled if pooled > 1e-12 else None
        else:
            effect_size = observed

        temporal: list[dict] = []
        first_folds = np.array_split(first_all, 3)
        second_folds = np.array_split(second_all, 3)
        for fold_number, (left, right) in enumerate(
            zip(first_folds, second_folds), start=1
        ):
            value = estimate(second, right) - estimate(first, left)
            temporal.append(
                {
                    "fold": fold_number,
                    "matches": int(len(left) + len(right)),
                    "value": None if not math.isfinite(value) else value,
                    "direction": (
                        0 if not math.isfinite(value) else _direction(value)
                    ),
                }
            )
        result = StatisticalResult(
            value=float(observed),
            baseline=float(first_effect),
            effect_size=(None if effect_size is None else float(effect_size)),
            confidence_interval_95=(float(ci_values[0]), float(ci_values[1])),
            p_value=float(p_value),
            events_by_group={
                first_patch: len(first.primary),
                second_patch: len(second.primary),
            },
        )
        return result, tuple(temporal)

    def _compare_patches(
        self,
        request: AnalysisRequest,
        question: str,
        cohort: ResolvedCohort | None,
    ) -> list[EvidencePacket]:
        patches = request.filters.patches
        if (
            request.metric is None
            or len(patches) != 2
            or len(set(patches)) != 2
        ):
            raise ValueError("UNSUPPORTED_FILTER: compare_patches")
        spec = get_metric(request.metric)
        resolved = cohort or self.resolver.resolve(request)
        dataset_version = self._dataset_version()
        cached = self._cached(dataset_version, request, question)
        if cached is not None:
            return [cached]

        patch_runs: list[_MetricRun] = []
        for patch in patches:
            patch_request = self._patch_request(request, patch)
            patch_cohort = self.resolver.resolve(patch_request)
            patch_runs.append(self._query_metric(patch_request, patch_cohort, spec))
        result, temporal = self._patch_contrast_result(
            patch_runs[0], patch_runs[1], patches[0], patches[1]
        )
        packet = build_evidence(
            question=question,
            request=request,
            cohort=resolved,
            metric_name=f"{spec.name}_patch_contrast",
            unit=spec.unit,
            result=result,
            q_value=result.p_value,
            temporal_folds=temporal,
            match_ids=patch_runs[0].match_ids + patch_runs[1].match_ids,
            dataset_version=dataset_version,
            method_name="patch_effect_contrast_bootstrap_permutation",
            family_size=1,
            config=self.config,
            group_counts=(len(patch_runs[0].primary), len(patch_runs[1].primary)),
            win_rate_difference=(
                result.value if spec.kind is MetricKind.BINARY else None
            ),
            additional_limitations=("PATCH_CONTRAST_RETROSPECTIVE",),
        )
        self._store(packet, dataset_version, request, question)
        return [packet]

    def _resolve_regional_cohort(
        self, request: AnalysisRequest
    ) -> ResolvedCohort:
        regions = request.filters.regions
        if len(regions) != 2 or len(set(regions)) != 2 or request.metric is None:
            raise ValueError("UNSUPPORTED_FILTER: compare_regions")
        if request.filters.patches:
            return self.resolver.resolve(request)

        with connect_analytics(self.path, readonly=True) as conn:
            available = [
                str(row[0])
                for row in conn.execute(
                    "SELECT patch FROM match_facts WHERE queue_id=420 "
                    "GROUP BY patch ORDER BY MAX(game_creation_ms) DESC LIMIT 3"
                )
            ]
        last_cohort: ResolvedCohort | None = None
        last_error: AnalyticsAbstention | None = None
        for size in range(1, len(available) + 1):
            patches = tuple(available[:size])
            candidate = AnalysisRequest(
                request.intent,
                request.metric,
                replace(request.filters, patches=patches),
            )
            try:
                cohort = self.resolver.resolve(candidate)
            except AnalyticsAbstention as exc:
                last_error = exc
                continue
            last_cohort = replace(cohort, widened=size > 1)
            region_counts = []
            for region in regions:
                region_request = AnalysisRequest(
                    request.intent,
                    request.metric,
                    replace(candidate.filters, regions=(region,)),
                )
                matches, _ = self.resolver._counts(region_request, patches)
                region_counts.append(matches)
            if min(region_counts) >= self.config.min_matches:
                return last_cohort
        if last_cohort is not None:
            return last_cohort
        if last_error is not None:
            raise last_error
        raise AnalyticsAbstention("INSUFFICIENT_MATCHES", {"matches": 0})

    def _regional_observations(
        self,
        request: AnalysisRequest,
        cohort: ResolvedCohort,
        spec: MetricSpec,
    ) -> tuple[list[RegionalObservation], list[str]]:
        where_sql, params = self.resolver.where_clause(request, cohort.patches)
        needs_players = (
            request.filters.champion is not None
            or request.filters.role is not None
        )
        player_join = (
            " JOIN player_facts p ON p.match_id=m.match_id" if needs_players else ""
        )
        column = spec.column
        with connect_analytics(self.path, readonly=True) as conn:
            if spec.kind is MetricKind.CONTINUOUS:
                rows = conn.execute(
                    "SELECT DISTINCT m.match_id,m.region,m.patch,"
                    "m.game_creation_ms,"
                    f"winner.{column}-loser.{column} AS metric_value "
                    "FROM match_facts m "
                    "JOIN team_facts winner ON winner.match_id=m.match_id "
                    "AND winner.win=1 "
                    "JOIN team_facts loser ON loser.match_id=m.match_id "
                    "AND loser.win=0"
                    + player_join
                    + where_sql
                    + " ORDER BY m.game_creation_ms,m.match_id",
                    params,
                ).fetchall()
                observations = [
                    RegionalObservation(
                        region=str(row["region"]),
                        patch=str(row["patch"]),
                        value=float(row["metric_value"]),
                        game_creation_ms=int(row["game_creation_ms"]),
                        outcome=None,
                    )
                    for row in rows
                ]
            else:
                rows = conn.execute(
                    "SELECT DISTINCT m.match_id,m.region,m.patch,"
                    "m.game_creation_ms,"
                    f"blue.{column} AS metric_value,blue.win AS outcome "
                    "FROM match_facts m "
                    "JOIN team_facts blue ON blue.match_id=m.match_id "
                    "AND blue.team_id=100"
                    + player_join
                    + where_sql
                    + " ORDER BY m.game_creation_ms,m.match_id",
                    params,
                ).fetchall()
                observations = [
                    RegionalObservation(
                        region=str(row["region"]),
                        patch=str(row["patch"]),
                        value=float(row["metric_value"]),
                        game_creation_ms=int(row["game_creation_ms"]),
                        outcome=int(row["outcome"]),
                    )
                    for row in rows
                ]
        return observations, [str(row["match_id"]) for row in rows]

    def _compare_regions(
        self,
        request: AnalysisRequest,
        question: str,
        cohort: ResolvedCohort | None,
    ) -> list[EvidencePacket]:
        if request.metric is None:
            raise AnalyticsAbstention(
                "UNSUPPORTED_FILTER", {"required": "metric"}
            )
        spec = get_metric(request.metric)
        resolved = cohort or self._resolve_regional_cohort(request)
        dataset_version = self._dataset_version()
        cached = self._cached(dataset_version, request, question)
        if cached is not None:
            return [cached]
        observations, match_ids = self._regional_observations(
            request, resolved, spec
        )
        comparison = compare_regions(
            observations,
            seed=self.config.seed,
            resamples=self.config.bootstrap_resamples,
        )
        regions = tuple(sorted(comparison.region_effects))
        result = StatisticalResult(
            value=comparison.raw_contrast,
            baseline=comparison.region_effects[regions[1]],
            effect_size=comparison.standardized_contrast,
            confidence_interval_95=comparison.confidence_interval_95,
            p_value=comparison.p_value,
            events_by_group=comparison.events_by_region,
        )
        fold_size = len(observations) // 3
        temporal_folds = tuple(
            {
                "fold": index + 1,
                "matches": (
                    fold_size if index < 2 else len(observations) - 2 * fold_size
                ),
                "value": None,
                "direction": direction,
            }
            for index, direction in enumerate(comparison.temporal_directions)
        )
        packet = build_evidence(
            question=question,
            request=request,
            cohort=resolved,
            metric_name=f"{spec.name}_region_contrast",
            unit=spec.unit,
            result=result,
            q_value=comparison.q_value,
            temporal_folds=temporal_folds,
            match_ids=match_ids,
            dataset_version=dataset_version,
            method_name="patch_standardized_region_interaction",
            family_size=1,
            config=self.config,
            group_counts=tuple(comparison.events_by_region.values()),
            win_rate_difference=(
                comparison.raw_contrast
                if spec.kind is MetricKind.BINARY
                else None
            ),
            additional_limitations=comparison.limitations,
        )
        recommendation = dict(packet.recommendation)
        allowed = recommendation["allowed"] and comparison.display_allowed
        if not allowed:
            recommendation = {
                "allowed": False,
                "strength": "none",
                "wording_key": "descriptive_only",
            }
        robustness = dict(packet.robustness)
        robustness["regional_effects"] = dict(comparison.region_effects)
        robustness["recent_direction_matches"] = (
            comparison.recent_direction_matches
        )
        robustness["passed"] = allowed
        limitations = list(packet.limitations)
        if not allowed and "EFFECT_NOT_ROBUST" not in limitations:
            limitations.append("EFFECT_NOT_ROBUST")
        packet = replace(
            packet,
            recommendation=recommendation,
            robustness=robustness,
            limitations=tuple(limitations),
        )
        self._store(packet, dataset_version, request, question)
        return [packet]

    def run(
        self,
        request: AnalysisRequest,
        question: str,
        cohort: ResolvedCohort | None = None,
    ) -> list[EvidencePacket]:
        if request.intent is Intent.COMPARE_PATCHES:
            return self._compare_patches(request, question, cohort)
        if request.intent is Intent.COMPARE_REGIONS:
            return self._compare_regions(request, question, cohort)
        if request.intent in {Intent.RANK_CONDITIONS, Intent.EXPLAIN_METRIC}:
            return self._standard_run(request, question, cohort)
        raise AnalyticsAbstention("UNSUPPORTED_INTENT")

    def cached(
        self,
        request: AnalysisRequest,
        question: str,
    ) -> list[EvidencePacket]:
        """Read a single cached evidence packet without executing analysis."""

        dataset_version = self._dataset_version()
        packet = self._cached(dataset_version, request, question)
        return [] if packet is None else [packet]

    def run_uncached(
        self,
        request: AnalysisRequest,
        question: str,
        cohort: ResolvedCohort | None = None,
    ) -> list[EvidencePacket]:
        """Execute a standard analysis while bypassing cache reads."""

        if request.intent not in {Intent.RANK_CONDITIONS, Intent.EXPLAIN_METRIC}:
            raise AnalyticsAbstention("UNSUPPORTED_INTENT")
        return self._standard_run(
            request,
            question,
            cohort,
            read_cache=False,
        )
