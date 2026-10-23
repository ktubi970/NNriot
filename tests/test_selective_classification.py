"""Unit tests for selective classification engine and Wilson score calibration.

Follows Chow (1970) optimal rejection rule and Wilson (1927) score confidence bounds.
"""
from typing import Sequence
import numpy as np
import pytest

from core.selective_classification import (
    compute_conviction,
    wilson_score_interval,
    evaluate_selective_coverage,
    find_threshold_for_accuracy,
    stratify_by_conviction,
    DEFAULT_CONVICTION_THRESHOLDS,
)


def test_compute_conviction_invariance():
    """Verify conviction computation and anti-symmetry invariance C(A, B) == C(B, A)."""
    prob = np.array([0.55, 0.45, 0.60, 0.40, 0.50])
    conv = compute_conviction(prob)
    np.testing.assert_allclose(conv, [0.05, 0.05, 0.10, 0.10, 0.0], atol=1e-6)

    # Invariance under team swap: P -> 1 - P gives identical conviction
    conv_swapped = compute_conviction(1.0 - prob)
    np.testing.assert_allclose(conv, conv_swapped, atol=1e-6)

    # Extremes
    extremes = np.array([0.0, 1.0, 0.5])
    np.testing.assert_allclose(compute_conviction(extremes), [0.5, 0.5, 0.0], atol=1e-6)

    # 2D column vector handling
    prob_2d = prob.reshape(-1, 1)
    conv_2d = compute_conviction(prob_2d)
    np.testing.assert_allclose(conv_2d.squeeze(), conv, atol=1e-6)


def test_wilson_score_interval():
    """Verify Wilson score confidence interval calculation."""
    # Standard sample: 60 successes out of 100
    lower, upper = wilson_score_interval(successes=60, total=100, confidence_level=0.95)
    assert 0.49 < lower < 0.60
    assert 0.60 < upper < 0.70
    assert lower < upper

    # Small sample: 2 successes out of 2
    lower, upper = wilson_score_interval(successes=2, total=2, confidence_level=0.95)
    assert lower > 0.0
    assert np.isclose(upper, 1.0, atol=1e-6)

    # Zero total count (abstain everything edge case)
    lower_zero, upper_zero = wilson_score_interval(successes=0, total=0, confidence_level=0.95)
    assert lower_zero == 0.0
    assert upper_zero == 0.0

    # 0 successes out of 10
    lower_fail, upper_fail = wilson_score_interval(successes=0, total=10, confidence_level=0.95)
    assert lower_fail == 0.0
    assert upper_fail > 0.0


def test_evaluate_selective_coverage_abstention():
    """Verify selective coverage evaluation and abstention logic."""
    # 4 predictions from task brief
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.70, 0.30, 0.51, 0.49])

    # At tau = 0.05, only the first 2 are retained (|0.70-0.5|=0.20, |0.30-0.5|=0.20)
    res = evaluate_selective_coverage(y_true, y_prob, threshold=0.05)
    assert res["count"] == 2
    assert res["coverage"] == 0.50
    assert res["accuracy"] == 1.0
    assert res["wilson_lower"] > 0.0
    assert res["wilson_upper"] == 1.0
    assert res["threshold"] == 0.05

    # At tau = 0.0, all 4 predictions retained
    res_all = evaluate_selective_coverage(y_true, y_prob, threshold=0.0)
    assert res_all["count"] == 4
    assert res_all["coverage"] == 1.0
    assert res_all["accuracy"] == 1.0

    # Test with genuine incorrect low-margin predictions
    y_true_with_errors = np.array([1, 0, 0, 1])
    # At tau = 0.05: retained are 2 high conviction correct ones
    res_err_05 = evaluate_selective_coverage(y_true_with_errors, y_prob, threshold=0.05)
    assert res_err_05["count"] == 2
    assert res_err_05["coverage"] == 0.50
    assert res_err_05["accuracy"] == 1.0
    # At tau = 0.0: all retained, 2 correct, 2 incorrect -> 50% accuracy
    res_err_00 = evaluate_selective_coverage(y_true_with_errors, y_prob, threshold=0.0)
    assert res_err_00["count"] == 4
    assert res_err_00["coverage"] == 1.0
    assert res_err_00["accuracy"] == 0.50

    # At tau = 0.45, zero retained (complete abstention)
    res_none = evaluate_selective_coverage(y_true, y_prob, threshold=0.45)
    assert res_none["count"] == 0
    assert res_none["coverage"] == 0.0
    assert res_none["accuracy"] == 0.0
    assert res_none["wilson_lower"] == 0.0
    assert res_none["wilson_upper"] == 0.0


def test_evaluate_selective_coverage_anti_symmetry():
    """Verify evaluation is invariant under team-swap."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.65, 0.35, 0.52, 0.48, 0.70, 0.40, 0.58, 0.80])

    res_orig = evaluate_selective_coverage(y_true, y_prob, threshold=0.04)
    # Swap teams: Y -> 1 - Y, P -> 1 - P
    res_swap = evaluate_selective_coverage(1 - y_true, 1.0 - y_prob, threshold=0.04)

    assert res_orig["count"] == res_swap["count"]
    assert res_orig["coverage"] == res_swap["coverage"]
    assert np.isclose(res_orig["accuracy"], res_swap["accuracy"])
    assert np.isclose(res_orig["wilson_lower"], res_swap["wilson_lower"])
    assert np.isclose(res_orig["wilson_upper"], res_swap["wilson_upper"])


def test_find_threshold_for_accuracy():
    """Verify adaptive threshold search for target accuracy."""
    np.random.seed(42)
    # Generate noisy predictions where high margin has higher accuracy
    n = 1000
    y_true = np.random.randint(0, 2, size=n)
    # Correlated probabilities
    y_prob = 0.5 + (y_true - 0.5) * np.random.uniform(0.0, 0.4, size=n)

    res = find_threshold_for_accuracy(y_true, y_prob, target_accuracy=0.65)
    assert res["threshold"] >= 0.0
    assert res["accuracy"] >= 0.64
    assert res["coverage"] > 0.0
    assert "count" in res
    assert "wilson_lower" in res


def test_find_threshold_noisy_gradient():
    """Verify threshold search on noisy data where only high conviction meets accuracy."""
    np.random.seed(123)
    n = 2000
    y_true = np.random.randint(0, 2, size=n)
    
    # 50% of samples are low margin (pure noise coinflips: conviction in [0.0, 0.03])
    # 50% of samples are high margin (predictive: conviction in [0.08, 0.25], 80% correct)
    noise_mask = np.random.rand(n) < 0.5
    y_prob = np.zeros(n, dtype=float)
    
    # Coin-flip noise samples
    y_prob[noise_mask] = 0.5 + np.random.uniform(-0.03, 0.03, size=np.sum(noise_mask))
    
    # Predictive samples
    high_margin_correct = np.random.rand(np.sum(~noise_mask)) < 0.80
    high_margin_signs = np.where(high_margin_correct, y_true[~noise_mask] * 2 - 1, (1 - y_true[~noise_mask]) * 2 - 1)
    y_prob[~noise_mask] = 0.5 + high_margin_signs * np.random.uniform(0.08, 0.25, size=np.sum(~noise_mask))
    y_prob = np.clip(y_prob, 0.01, 0.99)

    # Target 75% accuracy (requires filtering out low-margin coinflips)
    res = find_threshold_for_accuracy(y_true, y_prob, target_accuracy=0.75)
    assert res["threshold"] >= 0.02
    assert res["accuracy"] >= 0.75
    assert res["target_met"] is True


def test_find_threshold_unmet():
    """Verify fallback behavior when target accuracy is unachievable."""
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_prob = np.full(n, 0.9)  # 0% accuracy (predicts 1, truth is 0)

    res = find_threshold_for_accuracy(y_true, y_prob, target_accuracy=0.90)
    assert res["target_met"] is False
    assert res["accuracy"] == 0.0


def test_stratify_by_conviction():
    """Verify multi-tier conviction stratification matches ladder specifications."""
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    y_prob = np.array([0.65, 0.35, 0.52, 0.48, 0.70, 0.60, 0.40, 0.30])

    ladder = stratify_by_conviction(y_true, y_prob)

    # Check default keys
    expected_keys = [
        "conviction_ge_0pct",
        "conviction_ge_2pct",
        "conviction_ge_4pct",
        "conviction_ge_6pct",
        "conviction_ge_8pct",
        "conviction_ge_10pct",
        "conviction_ge_12pct",
        "conviction_ge_15pct",
    ]
    for key in expected_keys:
        assert key in ladder
        assert "coverage" in ladder[key]
        assert "accuracy" in ladder[key]
        assert "count" in ladder[key]
        assert "wilson_lower" in ladder[key]
        assert "wilson_upper" in ladder[key]
        assert "threshold" in ladder[key]

    # Monotonic coverage non-increasing
    coverages = [ladder[k]["coverage"] for k in expected_keys]
    assert all(c1 >= c2 for c1, c2 in zip(coverages[:-1], coverages[1:]))

    # Custom thresholds
    custom = stratify_by_conviction(y_true, y_prob, thresholds=[0.05, 0.10])
    assert "conviction_ge_5pct" in custom
    assert "conviction_ge_10pct" in custom
    assert len(custom) == 2


def test_compute_conviction_scalar_and_error():
    """Verify scalar input and error handling for NaN."""
    # Scalar float
    c = compute_conviction(0.62)
    assert np.isclose(c, 0.12)

    # Probabilities containing NaN should raise ValueError
    with pytest.raises(ValueError, match="NaN"):
        compute_conviction([0.5, np.nan, 0.6])

    # Probabilities outside [0, 1] beyond float tolerance should raise ValueError
    with pytest.raises(ValueError, match="Probabilities must be within"):
        compute_conviction([-0.05, 0.5])

    with pytest.raises(ValueError, match="Probabilities must be within"):
        compute_conviction([1.05, 0.5])

    # Slight float inaccuracies within 1e-5 are safely tolerated
    assert np.isclose(compute_conviction(-1e-6), 0.5)
    assert np.isclose(compute_conviction(1.0 + 1e-6), 0.5)


def test_wilson_score_interval_validation():
    """Verify input validation on Wilson score calculation."""
    with pytest.raises(ValueError, match="Invalid successes"):
        wilson_score_interval(successes=15, total=10)

    # Negative total
    assert wilson_score_interval(0, -5) == (0.0, 0.0)


def test_selective_coverage_mismatched_lengths_and_empty():
    """Verify error on length mismatch and clean handling of empty input."""
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_selective_coverage(y_true=[1, 0], y_prob=[0.6], threshold=0.0)

    empty_res = evaluate_selective_coverage(y_true=[], y_prob=[], threshold=0.04)
    assert empty_res["count"] == 0
    assert empty_res["coverage"] == 0.0
    assert empty_res["accuracy"] == 0.0
    assert empty_res["wilson_lower"] == 0.0
    assert empty_res["wilson_upper"] == 0.0


def test_selective_coverage_python_lists():
    """Verify functions accept Python standard lists as well as numpy arrays."""
    y_true = [1, 0, 1]
    y_prob = [0.8, 0.2, 0.35]
    res = evaluate_selective_coverage(y_true, y_prob, threshold=0.1)
    assert res["count"] == 3
    assert res["coverage"] == 1.0



@pytest.mark.parametrize(("probability", "threshold", "retained"), [
    (0.46, 0.04, 1),
    (0.54, 0.04, 1),
    (0.42, 0.08, 1),
    (0.58, 0.08, 1),
    (0.5399996, 0.04, 1),
    (0.5399994, 0.04, 0),
    (0.4600005, 0.04, 0),
    (0.5399995, 0.04, 0),
    (0.5799996, 0.08, 1),
    (0.5799994, 0.08, 0),
    (0.475, 0.025, 1),
    (0.525, 0.025, 1),
    (0.5000004, 0.000001, 0),
    (0.5000006, 0.000001, 1),
])
def test_selective_boundaries_match_serving_and_team_swap(probability, threshold, retained):
    """Metrics must evaluate the same selection rule that serves each match."""
    from core.selective_infer import classify_match_conviction

    label = int(probability > 0.5)
    for p, y in [(probability, label), (1.0 - probability, 1 - label)]:
        result = evaluate_selective_coverage([y], [p], threshold=threshold)
        serving = classify_match_conviction(p, action_threshold=threshold, high_threshold=threshold)
        assert result["count"] == retained
        assert result["coverage"] == float(retained)
        assert result["accuracy"] == float(retained)
        assert serving.is_actionable is bool(retained)
        assert (serving.prediction != "abstain") is bool(retained)


def test_selective_boundary_metrics_include_both_teams():
    """Both sides of each boundary contribute to counts and Wilson intervals."""
    probabilities = np.array([0.46, 0.54, 0.42, 0.58, 0.49, 0.51])
    labels = np.array([0, 0, 0, 1, 1, 0])
    for p, y in [(probabilities, labels), (1.0 - probabilities, 1 - labels)]:
        result = evaluate_selective_coverage(y, p, threshold=0.04)
        assert result["count"] == 4
        assert result["coverage"] == pytest.approx(2 / 3)
        assert result["accuracy"] == 0.75
        assert result["wilson_lower"] == pytest.approx(0.300641842582402)
        assert result["wilson_upper"] == pytest.approx(0.954412739190300)


def test_threshold_search_uses_served_boundary_selection():
    """An explicit boundary candidate is assessed using its served coverage."""
    result = find_threshold_for_accuracy(
        [0, 1], [0.46, 0.54], target_accuracy=1.0, min_coverage=1.0,
        candidate_thresholds=[0.04],
    )
    assert result["target_met"] is True
    assert result["threshold"] == 0.04
    assert result["count"] == 2
    assert result["coverage"] == 1.0


def test_exact_threshold_finds_narrow_nonmonotone_feasible_region():
    # Only discarding the first two errors meets .9007. Discarding more
    # correct predictions makes accuracy fall again, so binary search is unsafe.
    probabilities = .600001 + np.arange(1000) * 1e-6
    labels = np.ones(1000, dtype=int)
    labels[:2] = 0
    labels[-99:] = 0
    result = find_threshold_for_accuracy(labels, probabilities, target_accuracy=.9007)
    assert result["target_met"] is True
    assert result["threshold"] == .100003
    assert result["count"] == 998
    assert result["accuracy"] == pytest.approx(899 / 998)


@pytest.mark.parametrize("min_coverage", [0.0, 101 / 102])
def test_exact_threshold_keeps_maximum_feasible_coverage(min_coverage):
    probabilities = .600001 + np.arange(102) * 1e-6
    labels = np.ones(102, dtype=int)
    labels[0] = 0
    result = find_threshold_for_accuracy(
        labels, probabilities, target_accuracy=1.0, min_coverage=min_coverage,
    )
    assert result["target_met"] is True
    assert result["count"] == 101
    assert result["threshold"] == .100002


def test_exact_threshold_uses_rounded_group_boundaries():
    probabilities = np.r_[np.full(39, .6000005001), np.full(13, .6000024999)]
    labels = np.r_[np.zeros(39, dtype=int), np.ones(13, dtype=int)]
    result = find_threshold_for_accuracy(labels, probabilities, target_accuracy=1.0)
    assert result["target_met"] is True
    assert result["threshold"] == .100002
    assert result["count"] == 13
    assert result["accuracy"] == 1.0


def test_exact_threshold_never_splits_served_ties():
    result = find_threshold_for_accuracy([0, 1], [.6000001, .6000002], target_accuracy=1.0)
    assert result["target_met"] is False
    assert result["threshold"] == 0.0
    assert result["count"] == 2
    assert result["accuracy"] == .5


@pytest.mark.parametrize(("min_count", "min_coverage", "count", "met"), [
    (2, .5, 2, True),
    (3, .5, 4, False),
    (2, .500001, 4, False),
    (5, 0, 4, False),
])
def test_exact_threshold_constraints_and_no_eligible_fallback(min_count, min_coverage, count, met):
    result = find_threshold_for_accuracy(
        [0, 0, 1, 1], [.51, .51, .6, .6], target_accuracy=1.0,
        min_count=min_count, min_coverage=min_coverage,
    )
    assert result["count"] == count
    assert result["target_met"] is met


def test_exact_threshold_fallback_prefers_coverage_when_accuracy_ties():
    result = find_threshold_for_accuracy(
        [0, 0, 1, 0, 1, 0], [.51, .51, .6, .6, .7, .7], target_accuracy=.9,
    )
    assert result["target_met"] is False
    assert result["count"] == 4
    assert result["accuracy"] == .5
    assert result["threshold"] == .1


@pytest.mark.parametrize(("candidates", "threshold", "count", "met"), [
    ([.2, .04], .04, 2, True),
    ([.2], .2, 1, True),
    ([], 0.0, 3, False),
    ([.5], 0.0, 3, False),
])
def test_exact_threshold_preserves_explicit_candidate_contract(candidates, threshold, count, met):
    result = find_threshold_for_accuracy(
        [0, 1, 1], [.51, .54, .7], target_accuracy=1.0,
        candidate_thresholds=candidates,
    )
    assert result["threshold"] == threshold
    assert result["count"] == count
    assert result["target_met"] is met


def test_exact_threshold_matches_exhaustive_serving_masks():
    # Deliberately small oracle: independently enumerate candidate masks via
    # the public evaluator, without sorting/cumulative-count implementation.
    rng = np.random.default_rng(20261011)
    for _ in range(30):
        p = rng.choice([.2, .3999995, .46, .4600005, .54, .6000001, .6000002, .8], size=24)
        y = rng.integers(0, 2, size=len(p))
        target = float(rng.choice([.5, .7, .95]))
        minimum = int(rng.choice([1, 8, 25]))
        coverage = float(rng.choice([0, .25, .75]))
        thresholds = sorted({0.0, *(round(max(float(v), 1-float(v))-.5, 6) for v in p)})
        rows = [evaluate_selective_coverage(y, p, t, confidence_level=.9) for t in thresholds]
        eligible = [r for r in rows if r["count"] >= minimum and r["coverage"] >= coverage]
        feasible = [r for r in eligible if r["accuracy"] >= target]
        expected = feasible[0] if feasible else max(eligible, key=lambda r: r["accuracy"], default=rows[0])
        kwargs = dict(target_accuracy=target, min_count=minimum, min_coverage=coverage, confidence_level=.9)
        actual = find_threshold_for_accuracy(y, p, **kwargs)
        assert {key: actual[key] for key in expected} == expected
        assert actual["target_met"] is bool(feasible)
        assert find_threshold_for_accuracy(1-y, 1-p, **kwargs) == actual
        order = rng.permutation(len(p))
        assert find_threshold_for_accuracy(y[order], p[order], **kwargs) == actual


@pytest.mark.parametrize(("labels", "probabilities"), [([], [.6]), ([1], []), ([1], [.6, .7])])
def test_exact_threshold_rejects_misaligned_observations(labels, probabilities):
    with pytest.raises(ValueError, match="Length mismatch"):
        find_threshold_for_accuracy(labels, probabilities)
