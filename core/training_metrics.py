"""Winner evaluation against a prior learned exclusively from training labels."""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from core.selective_classification import stratify_by_conviction

EPS = 1e-12


def _validate_labels_and_probabilities(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)

    if labels.ndim != 1 or p.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")
    if labels.size == 0 or labels.shape != p.shape:
        raise ValueError("Labels and probabilities must have equal nonzero length.")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Winner labels must be finite binary values (0 or 1).")
    if (
        not np.isfinite(p).all()
        or (p < 0.0).any()
        or (p > 1.0).any()
    ):
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    return labels, p


def _quantile_groups(probabilities: np.ndarray, num_bins: int) -> list[np.ndarray]:
    if type(num_bins) is not int or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")
    count = min(num_bins, len(probabilities))
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, count + 1)[1:-1]))
    indices = np.searchsorted(edges, probabilities, side="left")
    return [mask for i in range(len(edges) + 1)
            if (mask := (indices == i)).any()]


def _calibration_slope(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return {
            "status": "unavailable",
            "slope": None,
            "intercept": None,
            "reason": "single_class",
        }

    if np.all(probabilities == probabilities[0]):
        return {
            "status": "unavailable",
            "slope": None,
            "intercept": None,
            "reason": "constant_predictions",
        }

    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    logits = np.log(clipped) - np.log1p(-clipped)

    if np.ptp(logits) == 0.0:
        return {
            "status": "unavailable",
            "slope": None,
            "intercept": None,
            "reason": "constant_predictions",
        }

    logits_0 = logits[labels == 0]
    logits_1 = logits[labels == 1]
    if float(np.min(logits_1)) > float(np.max(logits_0)) or float(np.min(logits_0)) > float(np.max(logits_1)):
        return {
            "status": "unavailable",
            "slope": None,
            "intercept": None,
            "reason": "separation",
        }

    X = logits.reshape(-1, 1)
    y = labels.astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("error", category=ConvergenceWarning)
        try:
            clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            clf.fit(X, y)
            slope = float(clf.coef_[0][0])
            intercept = float(clf.intercept_[0])
            if not (np.isfinite(slope) and np.isfinite(intercept)):
                return {
                    "status": "unavailable",
                    "slope": None,
                    "intercept": None,
                    "reason": "fit_failed",
                }
            return {
                "status": "estimated",
                "slope": slope,
                "intercept": intercept,
                "reason": None,
            }
        except (ConvergenceWarning, FloatingPointError, np.linalg.LinAlgError):
            return {
                "status": "unavailable",
                "slope": None,
                "intercept": None,
                "reason": "fit_failed",
            }
        except (ValueError, RuntimeError):
            return {
                "status": "unavailable",
                "slope": None,
                "intercept": None,
                "reason": "fit_failed",
            }


def compute_calibration_diagnostics(y_true, probabilities, *, num_bins: int = 10, selective: bool = True) -> dict:
    if type(num_bins) is not int or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")

    labels, p = _validate_labels_and_probabilities(y_true, probabilities)

    groups = _quantile_groups(p, num_bins)

    base_rate = float(labels.mean())
    uncertainty = float(base_rate * (1.0 - base_rate))
    reliability = 0.0
    resolution = 0.0
    quantile_ece = 0.0
    binned_predictions = np.empty_like(p, dtype=float)

    for mask in groups:
        weight = float(mask.mean())
        mean_p = float(p[mask].mean())
        mean_y = float(labels[mask].mean())
        reliability += weight * (mean_p - mean_y) ** 2
        resolution += weight * (mean_y - base_rate) ** 2
        quantile_ece += weight * abs(mean_p - mean_y)
        binned_predictions[mask] = mean_p

    binned_brier = float(np.mean((binned_predictions - labels) ** 2))
    raw_brier = float(np.mean((p - labels) ** 2))
    binning_residual = float(raw_brier - binned_brier)

    slope_info = _calibration_slope(labels, p)

    out = {
        "quantile_ece": float(quantile_ece),
        "binning": {
            "method": "quantile",
            "requested_bins": int(num_bins),
            "effective_bins": int(len(groups)),
        },
        "brier_decomposition": {
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty),
            "binned_brier": float(binned_brier),
            "binning_residual": float(binning_residual),
            "identity_target": "binned_probabilities",
        },
        "calibration_slope": slope_info,
    }
    if selective:
        out["selective_metrics"] = stratify_by_conviction(labels, p)
    return out


def _winner_scores(labels: np.ndarray, probabilities: np.ndarray, *, diagnostics: bool = False, selective: bool = True) -> dict:
    # Ten equal-width probability bins, including p=1 in the final bin.
    bins = np.minimum((probabilities * 10).astype(int), 9)
    calibration_error = 0.0
    for bin_index in range(10):
        selected = bins == bin_index
        if selected.any():
            calibration_error += float(selected.mean()) * abs(
                float(probabilities[selected].mean())
                - float(labels[selected].mean())
            )

    scores = {
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if np.unique(labels).size == 2 else None
        ),
        "calibration_error": float(calibration_error),
    }
    if diagnostics:
        scores["diagnostics"] = compute_calibration_diagnostics(labels, probabilities, selective=selective)
    return scores


def evaluate_winner(y_true, probabilities, train_positive_rate: float = 0.5, *, diagnostics: bool = False) -> dict:
    """Score P(winner=1) without fitting or calibrating on evaluation outcomes.

    Inputs must be equally sized, nonempty one-dimensional arrays. The baseline
    always predicts ``train_positive_rate``; its class is therefore the training
    majority, even if the evaluation majority differs. Class decisions use
    P(winner=1) >= 0.5, with ties assigned to class 1.

    ``calibration_error`` is descriptive ECE over ten equal-width bins of
    P(winner=1), not evidence that a probability calibrator has been fitted.
    ROC AUC is undefined (None) for a split with only one observed class.
    """
    labels, probabilities = _validate_labels_and_probabilities(y_true, probabilities)
    train_rate = np.asarray(train_positive_rate, dtype=float)

    if train_rate.ndim != 0 or not np.isfinite(train_rate) or not 0 <= train_rate <= 1:
        raise ValueError("Training positive rate must be a finite scalar between 0 and 1.")

    model = _winner_scores(labels, probabilities, diagnostics=diagnostics, selective=True)
    baseline = _winner_scores(
        labels, np.full(labels.shape, float(train_rate)), diagnostics=diagnostics, selective=False
    )
    return {
        "model": model,
        "baseline": baseline,
        "accuracy_lift": model["accuracy"] - baseline["accuracy"],
        "log_loss_improvement": baseline["log_loss"] - model["log_loss"],
    }
