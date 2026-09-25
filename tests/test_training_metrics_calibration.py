"""Tests for calibration diagnostics, brier decomposition, and calibration slope."""

import json
import warnings
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from core.training_metrics import (
    evaluate_winner,
    compute_calibration_diagnostics,
    _quantile_groups,
    _calibration_slope,
)


def test_brier_identity_is_explicitly_for_binned_predictions():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9])
    d = compute_calibration_diagnostics(y, p, num_bins=2)["brier_decomposition"]
    assert np.mean((p - y) ** 2) == pytest.approx(0.045)
    assert d["binned_brier"] == pytest.approx(0.04)
    assert d["binning_residual"] == pytest.approx(0.005)
    assert d["binned_brier"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"])


def test_opt_in_diagnostics_preserve_legacy_values():
    old = evaluate_winner([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6], 0.5)
    new = evaluate_winner([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6], 0.5, diagnostics=True)
    assert set(new) == set(old)
    for part in ("model", "baseline"):
        assert {k: new[part][k] for k in old[part]} == old[part]
        assert "diagnostics" in new[part]
    # Ensure serializable with allow_nan=False
    json.dumps(new, allow_nan=False)


def test_constant_predictions_have_unavailable_slope_not_ideal_slope():
    d = compute_calibration_diagnostics([0, 1] * 20, [0.5] * 40)
    assert d["binning"]["effective_bins"] == 1
    assert d["calibration_slope"]["slope"] is None
    assert d["calibration_slope"]["intercept"] is None
    assert d["calibration_slope"]["reason"] == "constant_predictions"


@pytest.mark.parametrize("label", [0, 1])
def test_diagnostics_single_class_labels(label):
    labels = [label] * 20
    probs = np.linspace(0.1, 0.9, 20)
    d = compute_calibration_diagnostics(labels, probs, num_bins=5)
    assert d["calibration_slope"]["status"] == "unavailable"
    assert d["calibration_slope"]["reason"] == "single_class"
    assert d["calibration_slope"]["slope"] is None
    assert d["calibration_slope"]["intercept"] is None
    brier_dec = d["brier_decomposition"]
    assert brier_dec["uncertainty"] == 0.0
    assert brier_dec["resolution"] == 0.0
    assert brier_dec["binned_brier"] == pytest.approx(brier_dec["reliability"])


def test_diagnostics_exact_zero_and_one_probabilities():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    d = compute_calibration_diagnostics(y, p, num_bins=2)
    assert d["quantile_ece"] == pytest.approx(0.0)
    assert d["brier_decomposition"]["binned_brier"] == pytest.approx(0.0)
    # Perfect separation means slope is unavailable with separation
    assert d["calibration_slope"]["status"] == "unavailable"
    assert d["calibration_slope"]["reason"] == "separation"


def test_diagnostics_tied_quantiles_reduce_effective_bins():
    y = [0, 0, 1, 1, 0, 1]
    p = [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]
    d = compute_calibration_diagnostics(y, p, num_bins=10)
    assert d["binning"]["requested_bins"] == 10
    assert d["binning"]["effective_bins"] == 2


def test_diagnostics_num_bins_one():
    y = [0, 1, 0, 1]
    p = [0.2, 0.4, 0.6, 0.8]
    d = compute_calibration_diagnostics(y, p, num_bins=1)
    assert d["binning"]["effective_bins"] == 1


@pytest.mark.parametrize("bad_bins", [0, -1, True, False, 2.5, "10", None])
def test_diagnostics_invalid_num_bins(bad_bins):
    with pytest.raises(ValueError, match="num_bins must be a positive integer"):
        compute_calibration_diagnostics([0, 1], [0.3, 0.7], num_bins=bad_bins)


@pytest.mark.parametrize(("labels", "probs"), [
    ([], []),
    ([0, 1], [0.5]),
    ([0, 1], [0.2, -0.01]),
    ([0, 1], [0.2, 1.01]),
    ([0, 1], [0.2, np.nan]),
    ([0, 1], [0.2, np.inf]),
    ([0, 0.5], [0.2, 0.8]),
    ([0, 2], [0.2, 0.8]),
    ([[0], [1]], [0.2, 0.8]),
])
def test_diagnostics_invalid_inputs(labels, probs):
    with pytest.raises(ValueError):
        compute_calibration_diagnostics(labels, probs)


def test_calibration_slope_separation():
    # Labels 0 have probs <= 0.3; labels 1 have probs >= 0.7 (strict separation)
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.7, 0.9])
    res = _calibration_slope(y, p)
    assert res["status"] == "unavailable"
    assert res["reason"] == "separation"
    assert res["slope"] is None
    assert res["intercept"] is None


def test_calibration_slope_convergence_warning_handled(monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def _warn_fit(self, X, y):
        warnings.warn("Failed to converge", ConvergenceWarning)
        self.coef_ = np.array([[100.0]])
        self.intercept_ = np.array([0.0])
        return self

    monkeypatch.setattr(LogisticRegression, "fit", _warn_fit)

    # Overlapping data that would otherwise attempt fitting
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    res = _calibration_slope(y, p)
    assert res["status"] == "unavailable"
    assert res["reason"] == "fit_failed"
    assert res["slope"] is None


def test_calibration_slope_fit_failure_handled(monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def _fail_fit(self, X, y):
        raise FloatingPointError("Numerical overflow in solver")

    monkeypatch.setattr(LogisticRegression, "fit", _fail_fit)

    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    res = _calibration_slope(y, p)
    assert res["status"] == "unavailable"
    assert res["reason"] == "fit_failed"
    assert res["slope"] is None


def test_calibration_slope_successful_estimation():
    # Overlapping probabilities, well behaved
    np.random.seed(42)
    n = 200
    logits = np.random.normal(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-logits))
    # Generate labels roughly following probabilities
    y = (np.random.uniform(0, 1, n) < p).astype(int)

    d = compute_calibration_diagnostics(y, p, num_bins=10)
    slope_info = d["calibration_slope"]
    assert slope_info["status"] == "estimated"
    assert slope_info["reason"] is None
    assert isinstance(slope_info["slope"], float)
    assert isinstance(slope_info["intercept"], float)
    assert np.isfinite(slope_info["slope"])
    assert np.isfinite(slope_info["intercept"])
    assert 0.5 <= slope_info["slope"] <= 2.0
