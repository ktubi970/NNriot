"""Selective classification engine and Wilson score calibration.

Implements Chow's (1970) optimal rule for classification with a rejection option,
computing perspective-invariant conviction C = |P - 0.5|, Wilson score binomial confidence
intervals for selective accuracy, adaptive threshold optimization, and multi-tier
conviction stratification.

Academic references:
- Chow (1970): An Optimum Character Recognition System Using Decision Functions
- Wilson (1927): Probable Inference, the Law of Succession, and Statistical Inference
- Geifman & El-Yaniv (NeurIPS 2017): Selective Classification for Deep Neural Networks
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple
import numpy as np
from scipy import stats


DEFAULT_CONVICTION_THRESHOLDS: Tuple[float, ...] = (
    0.0,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15,
)


def compute_conviction(probabilities: np.ndarray | Sequence[float] | float) -> np.ndarray:
    """Compute conviction metric C = |P - 0.5|.

    Changing perspective on the same match replaces P with 1 - P and leaves
    conviction unchanged: |(1 - P) - 0.5| = |P - 0.5|. This does not require
    equal conviction after swapping compositions between fixed Blue/Red sides:
    the winner head retains its side bias under that different transformation.

    Args:
        probabilities: Predicted probabilities in [0, 1].

    Returns:
        np.ndarray of conviction values in [0.0, 0.5].
    """
    prob_arr = np.asarray(probabilities, dtype=float)
    if np.any(np.isnan(prob_arr)):
        raise ValueError("Probabilities contain NaN values")
    if np.any((prob_arr < -1e-5) | (prob_arr > 1.0 + 1e-5)):
        raise ValueError("Probabilities must be within [0, 1]")
    prob_arr = np.clip(prob_arr, 0.0, 1.0)
    # Canonicalize the favored side before subtraction. Although abs(p - .5)
    # is algebraically identical, its last bit can differ for p and 1-p and
    # change a decision at a decimal rounding boundary.
    return np.maximum(prob_arr, 1.0 - prob_arr) - 0.5


def _rounded_conviction(conviction: np.ndarray | Sequence[float] | float) -> np.ndarray:
    """Quantize canonical scores exactly as the serving decision does."""
    values = np.asarray(conviction, dtype=float)
    return np.fromiter(
        (round(float(value), 6) for value in values.flat),
        dtype=float,
        count=values.size,
    ).reshape(values.shape)


def conviction_at_least(
    conviction: np.ndarray | Sequence[float] | float,
    threshold: float,
) -> np.ndarray | np.bool_:
    """Apply the shared inclusive selection rule to canonical conviction.

    Use compute_conviction first. Decisions use the serving precision of six
    decimal places and the existing 1e-9 threshold tolerance. Python's round
    is intentional: NumPy rounding can differ at half-unit boundaries.
    The input score is not modified or used to recalibrate probabilities.
    """
    return _rounded_conviction(conviction) >= threshold - 1e-9


def wilson_score_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Calculate the Wilson score binomial confidence interval for proportion.

    Given by:
        (p + z^2 / (2n) +- z * sqrt(p*(1-p)/n + z^2 / (4n^2))) / (1 + z^2 / n)

    Guarantees valid confidence bounds even for extreme probabilities (p=0, p=1)
    and small sample sizes where the normal approximation breaks down.

    Args:
        successes: Number of successful outcomes (e.g. correct predictions).
        total: Total number of trials (e.g. retained samples).
        confidence_level: Confidence level, defaults to 0.95.

    Returns:
        Tuple of (lower_bound, upper_bound) in [0.0, 1.0].
    """
    if total <= 0:
        return 0.0, 0.0

    if successes < 0 or successes > total:
        raise ValueError(f"Invalid successes={successes} for total={total}")

    p = float(successes) / float(total)
    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    n = float(total)

    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    variance_term = (p * (1.0 - p)) / n + z2 / (4.0 * n * n)
    radius = z * np.sqrt(max(0.0, variance_term)) / denominator

    lower = max(0.0, center - radius)
    upper = min(1.0, center + radius)
    return float(lower), float(upper)


def evaluate_selective_coverage(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    threshold: float,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate accuracy, coverage, and Wilson bounds at a conviction threshold.

    Retains predictions using the same rounded conviction and inclusive
    threshold tolerance as serving (see conviction_at_least).
    Predictions are made via (y_prob >= 0.5).

    Args:
        y_true: Ground truth binary labels {0, 1}.
        y_prob: Predicted probabilities for class 1 in [0, 1].
        threshold: Conviction threshold tau >= 0.0.
        confidence_level: Confidence level for Wilson interval (default 0.95).

    Returns:
        Dict with keys:
            - threshold: float
            - coverage: float (retained / total)
            - accuracy: float (correct / retained, or 0.0 if retained == 0)
            - count: int (retained sample count)
            - wilson_lower: float
            - wilson_upper: float
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} items but y_prob has {len(y_prob_arr)}"
        )

    total_n = len(y_true_arr)
    threshold = float(threshold)

    if total_n == 0:
        return {
            "threshold": threshold,
            "coverage": 0.0,
            "accuracy": 0.0,
            "count": 0,
            "wilson_lower": 0.0,
            "wilson_upper": 0.0,
        }

    conviction = compute_conviction(y_prob_arr)
    retained_mask = conviction_at_least(conviction, threshold)
    retained_count = int(np.sum(retained_mask))

    if retained_count == 0:
        return {
            "threshold": threshold,
            "coverage": 0.0,
            "accuracy": 0.0,
            "count": 0,
            "wilson_lower": 0.0,
            "wilson_upper": 0.0,
        }

    retained_true = y_true_arr[retained_mask]
    retained_pred = (y_prob_arr[retained_mask] >= 0.5).astype(int)
    correct_count = int(np.sum(retained_pred == retained_true))

    coverage = float(retained_count) / float(total_n)
    accuracy = float(correct_count) / float(retained_count)
    wilson_lower, wilson_upper = wilson_score_interval(
        successes=correct_count,
        total=retained_count,
        confidence_level=confidence_level,
    )

    return {
        "threshold": threshold,
        "coverage": coverage,
        "accuracy": accuracy,
        "count": retained_count,
        "wilson_lower": wilson_lower,
        "wilson_upper": wilson_upper,
    }


def find_threshold_for_accuracy(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    target_accuracy: float = 0.61,
    min_coverage: float = 0.0,
    min_count: int = 1,
    candidate_thresholds: Sequence[float] | None = None,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Maximize empirical coverage subject to a target empirical accuracy.

    By default, search every distinct nonempty mask realizable by the serving
    rule, keeping rounded score ties together. Return a canonical threshold
    (zero for full coverage, otherwise an observed rounded conviction).
    Accuracy need not be monotone in the threshold. Sorting and cumulative
    counts take O(n log n) time and O(n) memory.

    If no eligible mask meets the target, return the eligible mask with highest
    accuracy, breaking ties by coverage. If none is eligible, return the full
    sample with target_met=False. Wilson intervals describe the selected sample;
    neither they nor target_met certify future accuracy after threshold search.

    Args:
        y_true: Ground truth binary labels {0, 1}.
        y_prob: Predicted probabilities for class 1 in [0, 1].
        target_accuracy: Minimum acceptable accuracy (e.g. 0.61 for 61%).
        min_coverage: Minimum coverage ratio required.
        min_count: Minimum number of samples required for evaluation.
        candidate_thresholds: Optional explicit list restricting the search.
        confidence_level: Confidence level for Wilson intervals.

    Returns:
        Dict containing threshold, accuracy, coverage, count, wilson bounds,
        and target_met boolean flag.
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} items but y_prob has {len(y_prob_arr)}"
        )

    if len(y_true_arr) == 0:
        return {
            "threshold": 0.0,
            "accuracy": 0.0,
            "coverage": 0.0,
            "count": 0,
            "wilson_lower": 0.0,
            "wilson_upper": 0.0,
            "target_met": False,
            "target_accuracy": float(target_accuracy),
        }

    conviction = compute_conviction(y_prob_arr)

    if candidate_thresholds is None:
        rounded = _rounded_conviction(conviction)
        order = np.argsort(rounded)
        scores = rounded[order]
        # A threshold cannot split observations with the same served score.
        starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
        counts = len(scores) - starts
        correct = ((y_prob_arr >= 0.5) == y_true_arr)[order]
        successes = np.cumsum(correct[::-1], dtype=np.int64)[::-1][starts]
        accuracies = successes / counts
        eligible = (counts >= min_count) & (counts / len(scores) >= min_coverage)
        feasible = np.flatnonzero(eligible & (accuracies >= target_accuracy))

        chosen = 0
        if feasible.size:
            chosen = int(feasible[0])
        elif np.any(eligible):
            eligible_indices = np.flatnonzero(eligible)
            chosen = int(eligible_indices[np.argmax(accuracies[eligible_indices])])

        # Consecutive rounded scores differ by at least 1e-6, so the serving
        # tolerance of 1e-9 cannot merge two distinct groups at these thresholds.
        threshold = 0.0 if chosen == 0 else float(scores[starts[chosen]])
        output = evaluate_selective_coverage(
            y_true_arr, y_prob_arr, threshold, confidence_level=confidence_level,
        )
        output["target_met"] = bool(feasible.size)
        output["target_accuracy"] = float(target_accuracy)
        return output

    candidates = np.sort(np.asarray(candidate_thresholds, dtype=float))

    best_candidate: Dict[str, Any] | None = None
    best_acc = -1.0

    for tau in candidates:
        res = evaluate_selective_coverage(
            y_true=y_true_arr,
            y_prob=y_prob_arr,
            threshold=float(tau),
            confidence_level=confidence_level,
        )

        if res["count"] >= min_count and res["coverage"] >= min_coverage:
            if res["accuracy"] > best_acc:
                best_acc = res["accuracy"]
                best_candidate = res

            if res["accuracy"] >= target_accuracy:
                output = dict(res)
                output["target_met"] = True
                output["target_accuracy"] = float(target_accuracy)
                return output

    # Fallback if target accuracy was not met by any threshold
    if best_candidate is not None:
        output = dict(best_candidate)
    else:
        output = evaluate_selective_coverage(
            y_true=y_true_arr,
            y_prob=y_prob_arr,
            threshold=0.0,
            confidence_level=confidence_level,
        )

    output["target_met"] = False
    output["target_accuracy"] = float(target_accuracy)
    return output


def format_conviction_key(threshold: float) -> str:
    """Format conviction threshold as standardized dictionary key.

    Examples:
        0.00 -> "conviction_ge_0pct"
        0.04 -> "conviction_ge_4pct"
        0.15 -> "conviction_ge_15pct"
        0.025 -> "conviction_ge_2_5pct"
    """
    pct = round(threshold * 100.0, 4)
    if abs(pct - round(pct)) < 1e-5:
        pct_str = f"{int(round(pct))}"
    else:
        pct_str = f"{pct:g}".replace(".", "_")
    return f"conviction_ge_{pct_str}pct"


def stratify_by_conviction(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    thresholds: Sequence[float] | None = None,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, Any]]:
    """Compute selective evaluation metrics across conviction ladder tiers.

    Default thresholds cover the full ladder:
    0% (unconditional baseline), 2%, 4%, 6%, 8%, 10%, 12%, 15%.

    Args:
        y_true: Ground truth binary labels {0, 1}.
        y_prob: Predicted probabilities for class 1 in [0, 1].
        thresholds: Optional sequence of conviction thresholds. Defaults to
            DEFAULT_CONVICTION_THRESHOLDS.
        confidence_level: Confidence level for Wilson intervals.

    Returns:
        Dict mapping tier keys (e.g. "conviction_ge_4pct") to evaluation dicts
        containing threshold, coverage, accuracy, count, and Wilson confidence bounds.
    """
    if thresholds is None:
        thresholds = DEFAULT_CONVICTION_THRESHOLDS

    strata: Dict[str, Dict[str, Any]] = {}
    for tau in thresholds:
        key = format_conviction_key(float(tau))
        strata[key] = evaluate_selective_coverage(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(tau),
            confidence_level=confidence_level,
        )

    return strata
