"""Evaluate an anti-symmetric ensemble blending Equivariant DeepSets with GBDT."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
from core.equivariant_model import ThreeClassEquivariantSoftmax
from core.experiment import read_manifest, restore_splits
from core.point_in_time_dataset import load_enriched_records
from core.deepsets_dataset import batch_extract_team_tensors
from core.enriched_features import batch_extract_enriched_features
from core.calibration import apply_temperature_to_probabilities, load_calibration_artifact
from core.training_metrics import evaluate_winner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_ensemble")


def evaluate_ensemble(
    model_path: str,
    db_path: str,
    samples: int = 45000,
) -> dict:
    checkpoint = Path(model_path).resolve()
    manifest_path = checkpoint.with_suffix(".split.json")
    calib_path = checkpoint.with_suffix(".calibration.json")
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = read_manifest(manifest_path)
    logger.info("Loading records from %s to match split manifest...", db_path)
    warmup = min(3000, max(500, samples // 3))
    records = load_enriched_records(db_path=db_path, limit=samples, warmup_matches=warmup)
    splits = restore_splits(records, manifest)
    
    logger.info("Restored splits: %s", {k: len(v) for k, v in splits.items()})

    # 1. Evaluate Equivariant DeepSets Neural Model
    logger.info("Evaluating Equivariant DeepSets model from %s...", checkpoint)
    custom_objects = {
        "SiameseTeamEncoder": SiameseTeamEncoder,
        "AntiSymmetricSoftmax": AntiSymmetricSoftmax,
        "ThreeClassEquivariantSoftmax": ThreeClassEquivariantSoftmax,
    }
    model = tf.keras.models.load_model(checkpoint, custom_objects=custom_objects)
    x_nn = {name: batch_extract_team_tensors([r["feature_json"] for r in rows])
            for name, rows in splits.items()}
    y = {name: np.asarray([r["labels_json"]["winner"] for r in rows], dtype=int)
         for name, rows in splits.items()}
    train_rate = float(y["train"].mean())

    nn_raw_probs = {}
    for name in ("train", "validation", "test"):
        preds = model.predict(x_nn[name], batch_size=128, verbose=0)
        p = np.asarray(preds["winner"])[:, 1]
        nn_raw_probs[name] = p

    # Apply calibration if available
    temp = 1.0
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        temp = float(calib_data.get("temperature", 1.0))
        logger.info("Loaded temperature calibration: T=%.4f", temp)
        nn_probs = {name: apply_temperature_to_probabilities(p, temp) for name, p in nn_raw_probs.items()}
    else:
        nn_probs = nn_raw_probs

    # 2. Train and Symmetrize Tree Models on Tabular Enriched Features
    logger.info("Extracting tabular enriched features for Tree Models...")
    x_tab = {name: batch_extract_enriched_features([r["feature_json"] for r in rows], mode="tabular")
             for name, rows in splits.items()}

    # Data augmentation: Team swap symmetrization (doubles training size and forces exact anti-symmetry)
    logger.info("Creating symmetrized training pairs for tree models...")
    swapped_train_features = [
        {"team_a": r["feature_json"]["team_b"], "team_b": r["feature_json"]["team_a"]}
        for r in splits["train"]
    ]
    x_tab_train_swapped = batch_extract_enriched_features(swapped_train_features, mode="tabular")
    x_tab_aug = np.concatenate([x_tab["train"], x_tab_train_swapped], axis=0)
    y_aug = np.concatenate([y["train"], 1 - y["train"]], axis=0)

    # Function to get strictly anti-symmetric predictions from any tree model: 0.5 * (P(B|A,B) + 1 - P(B|B,A))
    def _predict_symmetrized(model, split_name: str) -> np.ndarray:
        p1 = model.predict_proba(x_tab[split_name])[:, 1]
        swapped_feats = [
            {"team_a": r["feature_json"]["team_b"], "team_b": r["feature_json"]["team_a"]}
            for r in splits[split_name]
        ]
        x_swap = batch_extract_enriched_features(swapped_feats, mode="tabular")
        p2 = model.predict_proba(x_swap)[:, 1]
        return 0.5 * (p1 + (1.0 - p2))

    # A. HistGradientBoostingClassifier
    logger.info("Training HistGradientBoostingClassifier on symmetrized data (%d samples)...", len(y_aug))
    gbdt = HistGradientBoostingClassifier(
        categorical_features=list(range(10)),
        random_state=42,
        max_iter=200,
        early_stopping=True,
    )
    gbdt.fit(x_tab_aug, y_aug)
    gbdt_probs = {name: _predict_symmetrized(gbdt, name) for name in ("train", "validation", "test")}

    # B. LightGBM Classifier
    logger.info("Training LightGBM on symmetrized data (%d samples)...", len(y_aug))
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=45,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_model.fit(x_tab_aug, y_aug)
    lgb_probs = {name: _predict_symmetrized(lgb_model, name) for name in ("train", "validation", "test")}

    # 3. Optimize Blending Weights on Validation Split (Tri-blend: NN, LGBM, GBDT)
    val_y = y["validation"]
    best_weights = (1.0, 0.0, 0.0)
    best_val_log_loss = float("inf")

    # Grid search over simplex (w_nn + w_lgb + w_gbdt = 1.0)
    for w_nn in np.linspace(0.0, 1.0, 21):
        for w_lgb in np.linspace(0.0, 1.0 - w_nn, 21):
            w_gbdt = max(0.0, 1.0 - w_nn - w_lgb)
            blend_val = (w_nn * nn_probs["validation"] +
                         w_lgb * lgb_probs["validation"] +
                         w_gbdt * gbdt_probs["validation"])
            eval_res = evaluate_winner(val_y, blend_val, train_rate)
            if eval_res["model"]["log_loss"] < best_val_log_loss:
                best_val_log_loss = eval_res["model"]["log_loss"]
                best_weights = (float(w_nn), float(w_lgb), float(w_gbdt))

    logger.info("Optimal validation blending weights: w_nn=%.2f, w_lgb=%.2f, w_gbdt=%.2f (val_log_loss=%.4f)",
                best_weights[0], best_weights[1], best_weights[2], best_val_log_loss)

    # 4. Evaluate Test Metrics for all components and ensembles
    test_y = y["test"]
    nn_test_eval = evaluate_winner(test_y, nn_probs["test"], train_rate, diagnostics=True)
    gbdt_test_eval = evaluate_winner(test_y, gbdt_probs["test"], train_rate, diagnostics=True)
    lgb_test_eval = evaluate_winner(test_y, lgb_probs["test"], train_rate, diagnostics=True)

    w_nn, w_lgb, w_gbdt = best_weights
    ensemble_test_probs = (w_nn * nn_probs["test"] +
                           w_lgb * lgb_probs["test"] +
                           w_gbdt * gbdt_probs["test"])
    ensemble_test_eval = evaluate_winner(test_y, ensemble_test_probs, train_rate, diagnostics=True)

    # 2-way blend: DeepSets (0.65) + LightGBM (0.35)
    nn_lgb_probs = 0.65 * nn_probs["test"] + 0.35 * lgb_probs["test"]
    nn_lgb_eval = evaluate_winner(test_y, nn_lgb_probs, train_rate, diagnostics=True)

    # 5. Confidence Stratification (Accuracy by conviction margin)
    stratification = {}
    for margin in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
        mask = np.abs(ensemble_test_probs - 0.5) >= margin
        if mask.any():
            acc = float(np.mean((ensemble_test_probs[mask] >= 0.5) == test_y[mask]))
            stratification[f"conviction_ge_{int(margin*100)}pct"] = {
                "coverage_pct": round(float(mask.mean() * 100), 1),
                "count": int(mask.sum()),
                "accuracy": round(acc, 4),
            }

    report = {
        "split_counts": {k: len(v) for k, v in splits.items()},
        "best_weights": {"nn": w_nn, "lgb": w_lgb, "gbdt": w_gbdt},
        "deepsets_metrics": {
            "accuracy": nn_test_eval["model"]["accuracy"],
            "accuracy_lift": nn_test_eval["accuracy_lift"],
            "roc_auc": nn_test_eval["model"]["roc_auc"],
            "log_loss": nn_test_eval["model"]["log_loss"],
            "brier": nn_test_eval["model"]["brier"],
        },
        "lightgbm_metrics": {
            "accuracy": lgb_test_eval["model"]["accuracy"],
            "accuracy_lift": lgb_test_eval["accuracy_lift"],
            "roc_auc": lgb_test_eval["model"]["roc_auc"],
            "log_loss": lgb_test_eval["model"]["log_loss"],
            "brier": lgb_test_eval["model"]["brier"],
        },
        "gbdt_metrics": {
            "accuracy": gbdt_test_eval["model"]["accuracy"],
            "accuracy_lift": gbdt_test_eval["accuracy_lift"],
            "roc_auc": gbdt_test_eval["model"]["roc_auc"],
            "log_loss": gbdt_test_eval["model"]["log_loss"],
            "brier": gbdt_test_eval["model"]["brier"],
        },
        "deepsets_plus_lightgbm": {
            "accuracy": nn_lgb_eval["model"]["accuracy"],
            "accuracy_lift": nn_lgb_eval["accuracy_lift"],
            "roc_auc": nn_lgb_eval["model"]["roc_auc"],
            "log_loss": nn_lgb_eval["model"]["log_loss"],
            "brier": nn_lgb_eval["model"]["brier"],
        },
        "tri_blend_ensemble_metrics": {
            "accuracy": ensemble_test_eval["model"]["accuracy"],
            "accuracy_lift": ensemble_test_eval["accuracy_lift"],
            "roc_auc": ensemble_test_eval["model"]["roc_auc"],
            "log_loss": ensemble_test_eval["model"]["log_loss"],
            "brier": ensemble_test_eval["model"]["brier"],
            "calibration_error": ensemble_test_eval["model"]["calibration_error"],
        },
        "confidence_stratification": stratification,
        "baseline_accuracy": nn_test_eval["baseline"]["accuracy"],
    }

    out_file = checkpoint.with_name("ensemble_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved ensemble comparison report to %s", out_file)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--samples", type=int, default=45000)
    args = parser.parse_args()
    evaluate_ensemble(args.model, args.db, args.samples)
