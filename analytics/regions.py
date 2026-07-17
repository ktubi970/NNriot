"""Patch-adjusted, sensitivity-gated inter-region comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RegionalObservation:
    region: str
    patch: str
    value: float
    game_creation_ms: int
    outcome: int | None


@dataclass(frozen=True)
class RegionalComparison:
    region_effects: dict[str, float]
    standardized_contrast: float
    confidence_interval_95: tuple[float, float]
    p_value: float
    q_value: float
    display_allowed: bool
    limitations: tuple[str, ...]
    raw_contrast: float
    temporal_directions: tuple[int, ...]
    recent_direction_matches: bool
    events_by_region: dict[str, int]
    is_binary: bool


def _rate_difference(values: np.ndarray, outcomes: np.ndarray) -> float:
    exposed = outcomes[values == 1]
    comparison = outcomes[values == 0]
    if len(exposed) == 0 or len(comparison) == 0:
        return float("nan")
    return float(exposed.mean() - comparison.mean())


def _effect(values: np.ndarray, outcomes: np.ndarray | None) -> float:
    if outcomes is None:
        return float(values.mean())
    return _rate_difference(values, outcomes)


def _group_rows(
    rows: Iterable[RegionalObservation],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    outcomes: dict[tuple[str, str], list[int]] = defaultdict(list)
    binary = False
    for row in rows:
        key = (row.region, row.patch)
        values[key].append(float(row.value))
        if row.outcome is not None:
            binary = True
            outcomes[key].append(int(row.outcome))
    return {
        key: (
            np.asarray(cell_values, dtype=float),
            np.asarray(outcomes[key], dtype=np.int8) if binary else None,
        )
        for key, cell_values in values.items()
    }


def _patch_standardized(
    rows: list[RegionalObservation], regions: tuple[str, str]
) -> dict[str, float]:
    cells = _group_rows(rows)
    patches = sorted({row.patch for row in rows})
    patch_counts = {
        patch: sum(
            len(cells.get((region, patch), ((), None))[0]) for region in regions
        )
        for patch in patches
    }
    total = sum(patch_counts.values())
    effects: dict[str, float] = {}
    for region in regions:
        weighted = []
        for patch in patches:
            cell = cells.get((region, patch))
            if cell is None or len(cell[0]) == 0:
                continue
            value = _effect(*cell)
            if math.isfinite(value):
                weighted.append((patch_counts[patch] / total, value))
        denominator = sum(weight for weight, _ in weighted)
        effects[region] = (
            sum(weight * value for weight, value in weighted) / denominator
            if denominator
            else 0.0
        )
    return effects


def _contrast_direction(
    rows: list[RegionalObservation], regions: tuple[str, str]
) -> int:
    present = {row.region for row in rows}
    if set(regions) - present:
        return 0
    effects = _patch_standardized(rows, regions)
    contrast = effects[regions[0]] - effects[regions[1]]
    if contrast > 0:
        return 1
    if contrast < 0:
        return -1
    return 0


def regional_display_allowed(
    *,
    counts: dict[str, int],
    ci: tuple[float, float],
    q_value: float,
    standardized_contrast: float,
    raw_contrast: float,
    is_binary: bool,
    recent_direction_matches: bool,
    binary_powered: bool,
    common_patch_coverage: bool,
) -> bool:
    practical = (
        abs(raw_contrast) >= 0.03
        if is_binary
        else abs(standardized_contrast) >= 0.20
    )
    return (
        bool(counts)
        and min(counts.values()) >= 500
        and (ci[0] > 0 or ci[1] < 0)
    ) and (
        q_value <= 0.05
        and practical
        and recent_direction_matches
        and binary_powered
        and common_patch_coverage
    )


def compare_regions(
    rows: Iterable[RegionalObservation],
    seed: int,
    resamples: int = 2000,
) -> RegionalComparison:
    observations = list(rows)
    regions = tuple(sorted({row.region for row in observations}))
    if len(regions) != 2 or len(observations) < 4 or resamples < 1:
        raise ValueError("SOURCE_DATA_INVALID")
    if not all(math.isfinite(float(row.value)) for row in observations):
        raise ValueError("SOURCE_DATA_INVALID")
    outcome_flags = {row.outcome is None for row in observations}
    if len(outcome_flags) != 1:
        raise ValueError("SOURCE_DATA_INVALID")
    is_binary = observations[0].outcome is not None
    if is_binary and any(
        row.value not in (0, 1) or row.outcome not in (0, 1)
        for row in observations
    ):
        raise ValueError("SOURCE_DATA_INVALID")

    counts = Counter(row.region for row in observations)
    patches = sorted({row.patch for row in observations})
    cells = _group_rows(observations)
    common_patch_coverage = all(
        (region, patch) in cells for region in regions for patch in patches
    )
    binary_powered = True
    if is_binary:
        for region in regions:
            region_rows = [row for row in observations if row.region == region]
            exposure_counts = Counter(int(row.value) for row in region_rows)
            outcome_counts = Counter(int(row.outcome) for row in region_rows)
            if (
                min(exposure_counts.get(0, 0), exposure_counts.get(1, 0)) < 100
                or min(outcome_counts.get(0, 0), outcome_counts.get(1, 0)) < 100
            ):
                binary_powered = False
        for cell_values, cell_outcomes in cells.values():
            if not math.isfinite(_effect(cell_values, cell_outcomes)):
                binary_powered = False

    region_effects = _patch_standardized(observations, regions)
    raw_contrast = region_effects[regions[0]] - region_effects[regions[1]]
    if is_binary:
        standardized = raw_contrast
    else:
        region_values = {
            region: np.asarray(
                [row.value for row in observations if row.region == region],
                dtype=float,
            )
            for region in regions
        }
        numerator = sum(
            (len(values) - 1) * values.var(ddof=1)
            for values in region_values.values()
        )
        denominator = sum(len(values) for values in region_values.values()) - 2
        pooled = math.sqrt(float(numerator / denominator)) if denominator > 0 else 0.0
        standardized = raw_contrast / pooled if pooled > 1e-12 else 0.0

    limitations: list[str] = []
    if min(counts.values()) < 500:
        limitations.append("INSUFFICIENT_MATCHES")
    if not common_patch_coverage:
        limitations.append("PATCH_COVERAGE_MISMATCH")
    if not binary_powered:
        limitations.append("SPARSE_COMPARISON_GROUP")

    common_patches = [
        patch
        for patch in patches
        if all((region, patch) in cells for region in regions)
    ]
    if (is_binary and not binary_powered) or not common_patches:
        ci = (0.0, 0.0)
        p_value = 1.0
    else:
        patch_weights = {
            patch: sum(len(cells[(region, patch)][0]) for region in regions)
            / len(observations)
            for patch in common_patches
        }
        rng = np.random.default_rng(seed)
        bootstrap = np.empty(resamples, dtype=float)
        for draw in range(resamples):
            sampled_effects: dict[str, float] = {}
            for region in regions:
                value = 0.0
                weight_total = 0.0
                for patch, weight in patch_weights.items():
                    cell_values, cell_outcomes = cells[(region, patch)]
                    indices = rng.integers(0, len(cell_values), len(cell_values))
                    sampled = _effect(
                        cell_values[indices],
                        None if cell_outcomes is None else cell_outcomes[indices],
                    )
                    if math.isfinite(sampled):
                        value += weight * sampled
                        weight_total += weight
                sampled_effects[region] = value / weight_total
            bootstrap[draw] = (
                sampled_effects[regions[0]] - sampled_effects[regions[1]]
            )
        ci_values = np.quantile(bootstrap, [0.025, 0.975])
        ci = (float(ci_values[0]), float(ci_values[1]))

        permuted = np.empty(resamples, dtype=float)
        for draw in range(resamples):
            effects_by_region = {regions[0]: 0.0, regions[1]: 0.0}
            weights_by_region = {regions[0]: 0.0, regions[1]: 0.0}
            for patch, weight in patch_weights.items():
                first_cell = cells[(regions[0], patch)]
                second_cell = cells[(regions[1], patch)]
                combined_values = np.concatenate([first_cell[0], second_cell[0]])
                combined_outcomes = (
                    None
                    if first_cell[1] is None
                    else np.concatenate([first_cell[1], second_cell[1]])
                )
                order = rng.permutation(len(combined_values))
                split = len(first_cell[0])
                for region, indices in (
                    (regions[0], order[:split]),
                    (regions[1], order[split:]),
                ):
                    effect = _effect(
                        combined_values[indices],
                        None
                        if combined_outcomes is None
                        else combined_outcomes[indices],
                    )
                    if math.isfinite(effect):
                        effects_by_region[region] += weight * effect
                        weights_by_region[region] += weight
            left = effects_by_region[regions[0]] / weights_by_region[regions[0]]
            right = effects_by_region[regions[1]] / weights_by_region[regions[1]]
            permuted[draw] = abs(left - right)
        p_value = (1.0 + float(np.count_nonzero(permuted >= abs(raw_contrast)))) / (
            resamples + 1.0
        )

    q_value = float(p_value)
    ordered_times = np.asarray(
        [row.game_creation_ms for row in observations], dtype=np.int64
    )
    first_cut, second_cut = np.quantile(ordered_times, [1 / 3, 2 / 3])
    temporal_subsets = (
        [row for row in observations if row.game_creation_ms <= first_cut],
        [
            row
            for row in observations
            if first_cut < row.game_creation_ms <= second_cut
        ],
        [row for row in observations if row.game_creation_ms > second_cut],
    )
    temporal_directions = tuple(
        _contrast_direction(subset, regions) for subset in temporal_subsets
    )
    median_time = float(np.median(ordered_times))
    recent = [
        row for row in observations if row.game_creation_ms >= median_time
    ]
    full_direction = 1 if raw_contrast > 0 else -1 if raw_contrast < 0 else 0
    recent_direction = _contrast_direction(recent, regions)
    recent_matches = full_direction != 0 and recent_direction == full_direction

    if not recent_matches:
        limitations.append("TEMPORAL_DIRECTION_UNSTABLE")
    excludes_null = ci[0] > 0 or ci[1] < 0
    practical = (
        abs(raw_contrast) >= 0.03
        if is_binary
        else abs(standardized) >= 0.20
    )
    if q_value > 0.05 or not excludes_null or not practical:
        limitations.append("EFFECT_NOT_ROBUST")
    allowed = regional_display_allowed(
        counts=dict(counts),
        ci=ci,
        q_value=q_value,
        standardized_contrast=float(standardized),
        raw_contrast=float(raw_contrast),
        is_binary=is_binary,
        recent_direction_matches=recent_matches,
        binary_powered=binary_powered,
        common_patch_coverage=common_patch_coverage,
    )
    return RegionalComparison(
        region_effects=region_effects,
        standardized_contrast=float(standardized),
        confidence_interval_95=ci,
        p_value=float(p_value),
        q_value=q_value,
        display_allowed=allowed,
        limitations=tuple(dict.fromkeys(limitations)),
        raw_contrast=float(raw_contrast),
        temporal_directions=temporal_directions,
        recent_direction_matches=recent_matches,
        events_by_region={region: counts[region] for region in regions},
        is_binary=is_binary,
    )
