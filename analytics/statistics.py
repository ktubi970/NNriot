"""Reproducible statistical primitives for analytical evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import fisher_exact, norm


@dataclass(frozen=True)
class StatisticalResult:
    value: float
    baseline: float
    effect_size: float | None
    confidence_interval_95: tuple[float, float]
    p_value: float
    events_by_group: dict[str, int]
    odds_ratio: float | None = None


def continuous_effect(
    differences: np.ndarray,
    seed: int,
    resamples: int = 2000,
) -> StatisticalResult:
    """Estimate a paired mean difference with bootstrap CI and sign test."""

    values = np.asarray(differences, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or not isinstance(resamples, int)
        or resamples < 1
    ):
        raise ValueError("SOURCE_DATA_INVALID")

    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    permuted = np.empty(resamples, dtype=float)
    chunk_size = 128
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        size = stop - start
        indices = rng.integers(
            0,
            len(values),
            size=(size, len(values)),
            dtype=np.int32,
        )
        draws[start:stop] = values[indices].mean(axis=1)
        signs = rng.integers(
            0,
            2,
            size=(size, len(values)),
            dtype=np.int8,
        )
        signs = signs * 2 - 1
        permuted[start:stop] = np.abs((signs * values).mean(axis=1))

    ci_values = np.quantile(draws, [0.025, 0.975])
    ci = (float(ci_values[0]), float(ci_values[1]))
    mean = float(values.mean())
    observed = abs(mean)
    p_value = (1.0 + float(np.count_nonzero(permuted >= observed))) / (
        resamples + 1.0
    )
    standard_deviation = float(values.std(ddof=1))
    dz = mean / standard_deviation if standard_deviation > 1e-12 else None
    return StatisticalResult(
        value=mean,
        baseline=0.0,
        effect_size=dz,
        confidence_interval_95=ci,
        p_value=float(p_value),
        events_by_group={"matches": len(values)},
    )


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = float(norm.ppf(0.975))
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            probability * (1 - probability) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return float(center - half), float(center + half)


def _binary_vector(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    try:
        finite = np.isfinite(raw).all()
    except TypeError as exc:
        raise ValueError("SOURCE_DATA_INVALID") from exc
    if raw.ndim != 1 or len(raw) == 0 or not finite:
        raise ValueError("SOURCE_DATA_INVALID")
    if not set(np.unique(raw).tolist()) <= {0, 1}:
        raise ValueError("SOURCE_DATA_INVALID")
    return raw.astype(np.int8, copy=False)


def binary_effect(
    exposure: np.ndarray,
    outcome: np.ndarray,
    min_group: int = 100,
) -> StatisticalResult:
    """Compare outcome rates for one binary exposure row per match."""

    x = _binary_vector(exposure)
    y = _binary_vector(outcome)
    if x.shape != y.shape or not isinstance(min_group, int) or min_group < 1:
        raise ValueError("SOURCE_DATA_INVALID")
    exposed = y[x == 1]
    comparison = y[x == 0]
    if min(len(exposed), len(comparison)) < min_group:
        raise ValueError("SPARSE_COMPARISON_GROUP")

    exposed_wins = int(exposed.sum())
    exposed_losses = int(len(exposed) - exposed_wins)
    comparison_wins = int(comparison.sum())
    comparison_losses = int(len(comparison) - comparison_wins)
    exposed_ci = _wilson(exposed_wins, len(exposed))
    comparison_ci = _wilson(comparison_wins, len(comparison))
    difference = exposed_wins / len(exposed) - comparison_wins / len(comparison)
    newcombe = (
        exposed_ci[0] - comparison_ci[1],
        exposed_ci[1] - comparison_ci[0],
    )
    odds_ratio, p_value = fisher_exact(
        [
            [exposed_wins, exposed_losses],
            [comparison_wins, comparison_losses],
        ],
        alternative="two-sided",
    )
    numeric_odds = float(odds_ratio)
    return StatisticalResult(
        value=float(difference),
        baseline=float(comparison_wins / len(comparison)),
        effect_size=float(difference),
        confidence_interval_95=(float(newcombe[0]), float(newcombe[1])),
        p_value=float(p_value),
        events_by_group={
            "exposed": len(exposed),
            "comparison": len(comparison),
        },
        odds_ratio=numeric_odds if math.isfinite(numeric_odds) else None,
    )


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return monotone BH-adjusted q-values in the original order."""

    try:
        values = np.asarray(p_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("SOURCE_DATA_INVALID") from exc
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("SOURCE_DATA_INVALID")
    count = len(values)
    if count == 0:
        return []

    order = sorted(range(count), key=values.__getitem__)
    adjusted = [1.0] * count
    running = 1.0
    for rank_from_end, index in enumerate(reversed(order), start=1):
        rank = count - rank_from_end + 1
        running = min(running, float(values[index]) * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted
