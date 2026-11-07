import math

import numpy as np
import pytest


def test_perfect_predictions_improve_over_a_chance_baseline():
    from core.training_metrics import evaluate_winner

    result = evaluate_winner([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0], 0.5)

    assert result["model"] == pytest.approx({
        "accuracy": 1.0,
        "log_loss": 0.0,
        "brier": 0.0,
        "roc_auc": 1.0,
        "calibration_error": 0.0,
    }, abs=1e-12)
    assert result["baseline"] == pytest.approx({
        "accuracy": 0.5,
        "log_loss": math.log(2),
        "brier": 0.25,
        "roc_auc": 0.5,
        "calibration_error": 0.0,
    })
    assert result["accuracy_lift"] == pytest.approx(0.5)
    assert result["log_loss_improvement"] == pytest.approx(math.log(2))


def test_chance_predictions_have_no_lift_over_matching_train_prior():
    from core.training_metrics import evaluate_winner

    result = evaluate_winner([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], 0.5)

    assert result["model"] == pytest.approx({
        "accuracy": 0.5,
        "log_loss": math.log(2),
        "brier": 0.25,
        "roc_auc": 0.5,
        "calibration_error": 0.0,
    })
    assert result["accuracy_lift"] == pytest.approx(0.0)
    assert result["log_loss_improvement"] == pytest.approx(0.0)


def test_baseline_keeps_train_prevalence_when_evaluation_majority_changes():
    from core.training_metrics import evaluate_winner

    result = evaluate_winner([0, 1, 1, 1], [0.6, 0.6, 0.6, 0.6], 0.2)

    assert result["baseline"] == pytest.approx({
        "accuracy": 0.25,
        "log_loss": 1.2628643221541276,
        "brier": 0.49,
        "roc_auc": 0.5,
        "calibration_error": 0.55,
    })
    assert result["model"]["accuracy"] == pytest.approx(0.75)
    assert result["accuracy_lift"] == pytest.approx(0.5)


def test_calibration_error_weights_nonempty_probability_bins_by_sample_count():
    from core.training_metrics import evaluate_winner

    result = evaluate_winner([0, 1, 0, 1], [0.01, 0.09, 0.21, 1.0], 0.5)

    # First bin: |0.05 - 0.5| * 2/4; third bin: |0.21 - 0| * 1/4.
    assert result["model"]["calibration_error"] == pytest.approx(0.2775)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_has_defined_losses_but_no_roc_auc(label):
    from core.training_metrics import evaluate_winner

    result = evaluate_winner([label, label], [0.25, 0.75], float(label))

    assert result["model"]["accuracy"] == pytest.approx(0.5)
    assert result["model"]["log_loss"] == pytest.approx(0.8369882167858358)
    assert result["model"]["brier"] == pytest.approx(0.3125)
    assert result["model"]["roc_auc"] is None
    assert result["baseline"]["roc_auc"] is None
    assert result["baseline"]["accuracy"] == pytest.approx(1.0)
    assert result["baseline"]["log_loss"] == pytest.approx(0.0, abs=1e-12)
    assert result["baseline"]["brier"] == pytest.approx(0.0)


@pytest.mark.parametrize(("labels", "probabilities", "train_rate"), [
    ([], [], 0.5),
    ([0, 1], [0.5], 0.5),
    ([[0], [1]], [0.2, 0.8], 0.5),
    ([0, 1], [[0.2], [0.8]], 0.5),
    ([0, 2], [0.2, 0.8], 0.5),
    ([0, 0.5], [0.2, 0.8], 0.5),
    ([0, np.nan], [0.2, 0.8], 0.5),
    ([0, np.inf], [0.2, 0.8], 0.5),
    ([0, 1], [0.2, np.nan], 0.5),
    ([0, 1], [0.2, np.inf], 0.5),
    ([0, 1], [-0.01, 0.8], 0.5),
    ([0, 1], [0.2, 1.01], 0.5),
    ([0, 1], [0.2, 0.8], np.nan),
    ([0, 1], [0.2, 0.8], np.inf),
    ([0, 1], [0.2, 0.8], -0.01),
    ([0, 1], [0.2, 0.8], 1.01),
    ([0, 1], [0.2, 0.8], [0.5]),
])
def test_invalid_evaluation_inputs_are_rejected(labels, probabilities, train_rate):
    from core.training_metrics import evaluate_winner

    with pytest.raises(ValueError):
        evaluate_winner(labels, probabilities, train_rate)
