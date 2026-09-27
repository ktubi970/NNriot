"""Evaluate the trained DeepSets v2 model checkpoint on test data with calibration and conviction stratification."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
import sys
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.features.matchup_matrix import LaneMatchupTracker
from core.features.pipeline import DraftFeaturePipeline, ENRICHED_PLAYER_TENSOR_DIM
from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
from core.equivariant_model import ThreeClassEquivariantSoftmax
from core.training_metrics import evaluate_winner
from core.calibration import fit_temperature, apply_temperature_to_probabilities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_deepsets_v2")


def main():
    model_dir = Path("runs/candidate_deepsets_v2").resolve()
    checkpoint = model_dir / "model.keras"
    splits_cache = Path("runs/candidate_selective_prod/splits_cache_45000.pkl").resolve()

    logger.info("Loading cached splits from %s...", splits_cache)
    with open(splits_cache, "rb") as f:
        splits = pickle.load(f)

    logger.info("Fitting LaneMatchupTracker on train split...")
    matchup_tracker = LaneMatchupTracker(matchup_weight=5.0)
    for r in splits["train"]:
        fj = r["feature_json"]
        w = r["labels_json"]["winner"]
        matchup_tracker.update_match(fj.get("team_a") or [], fj.get("team_b") or [], winner=w)

    pipeline = DraftFeaturePipeline(matchup_tracker=matchup_tracker)

    logger.info("Extracting tensors for validation and test splits...")
    x_val = pipeline.batch_extract_tensors([r["feature_json"] for r in splits["validation"]])
    x_test = pipeline.batch_extract_tensors([r["feature_json"] for r in splits["test"]])

    y_val = np.asarray([r["labels_json"]["winner"] for r in splits["validation"]], dtype=int)
    y_test = np.asarray([r["labels_json"]["winner"] for r in splits["test"]], dtype=int)
    y_train = np.asarray([r["labels_json"]["winner"] for r in splits["train"]], dtype=int)
    train_rate = float(y_train.mean())

    logger.info("Loading model from %s...", checkpoint)
    custom_objects = {
        "SiameseTeamEncoder": SiameseTeamEncoder,
        "AntiSymmetricSoftmax": AntiSymmetricSoftmax,
        "ThreeClassEquivariantSoftmax": ThreeClassEquivariantSoftmax,
    }
    model = tf.keras.models.load_model(checkpoint, custom_objects=custom_objects)

    # Inferences
    logger.info("Predicting validation set...")
    val_preds = model.predict(x_val, batch_size=128, verbose=0)
    raw_val_probs = np.asarray(val_preds["winner"])[:, 1]

    fit_res = fit_temperature(raw_val_probs, y_val)
    fitted_temp = float(fit_res["temperature"]) if isinstance(fit_res, dict) else float(fit_res)
    logger.info("Fitted temperature: T = %.4f (before log_loss=%.4f, after log_loss=%.4f)",
                fitted_temp, fit_res.get("log_loss_before", 0), fit_res.get("log_loss_after", 0))

    calib_file = model_dir / "model.calibration.json"
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump({"temperature": fitted_temp, "diagnostics": fit_res}, f, indent=2)

    logger.info("Predicting test set...")
    test_preds = model.predict(x_test, batch_size=128, verbose=0)
    raw_test_probs = np.asarray(test_preds["winner"])[:, 1]
    cal_test_probs = apply_temperature_to_probabilities(raw_test_probs, fitted_temp)

    eval_raw = evaluate_winner(y_test, raw_test_probs, train_rate, diagnostics=True)
    eval_cal = evaluate_winner(y_test, cal_test_probs, train_rate, diagnostics=True)

    stratification = {}
    for margin in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
        mask = np.abs(cal_test_probs - 0.5) >= margin
        if mask.any():
            acc = float(np.mean((cal_test_probs[mask] >= 0.5) == y_test[mask]))
            stratification[f"conviction_ge_{int(margin*100)}pct"] = {
                "coverage_pct": round(float(mask.mean() * 100), 1),
                "count": int(mask.sum()),
                "accuracy": round(acc, 4),
            }

    report = {
        "checkpoint": str(checkpoint),
        "temperature": fitted_temp,
        "test_metrics_raw": {
            "accuracy": eval_raw["model"]["accuracy"],
            "accuracy_lift": eval_raw["accuracy_lift"],
            "roc_auc": eval_raw["model"]["roc_auc"],
            "log_loss": eval_raw["model"]["log_loss"],
            "brier": eval_raw["model"]["brier"],
        },
        "test_metrics_calibrated": {
            "accuracy": eval_cal["model"]["accuracy"],
            "accuracy_lift": eval_cal["accuracy_lift"],
            "roc_auc": eval_cal["model"]["roc_auc"],
            "log_loss": eval_cal["model"]["log_loss"],
            "brier": eval_cal["model"]["brier"],
            "calibration_error": eval_cal["model"]["calibration_error"],
        },
        "confidence_stratification": stratification,
        "baseline_accuracy": eval_raw["baseline"]["accuracy"],
    }

    metrics_file = model_dir / "model.metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== FINAL TEST EVALUATION REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
