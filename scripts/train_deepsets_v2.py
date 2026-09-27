"""Train Equivariant DeepSets v2 with 14-dim player tensors and cross-lane attention."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.features.matchup_matrix import LaneMatchupTracker
from core.features.pipeline import DraftFeaturePipeline, ENRICHED_PLAYER_TENSOR_DIM
from core.equivariant_model import build_equivariant_deepsets_model
from core.generate_graph import LOSS_WEIGHTS, LOSS_PER_HEAD
from core.training_utils import _build_targets
from core.pcgrad import pcgrad_step
from core.training_metrics import evaluate_winner
from core.calibration import fit_temperature, apply_temperature_to_probabilities, save_calibration_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_deepsets_v2")


def train_deepsets_v2(
    splits_cache: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
    learning_rate: float = 0.0008,
) -> dict:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path / "model.keras"

    logger.info("Loading cached splits from %s...", splits_cache)
    with open(splits_cache, "rb") as f:
        splits = pickle.load(f)

    logger.info("Loaded splits: %s", {k: len(v) for k, v in splits.items()})

    # 1. Build Bayesian smoothed LaneMatchupTracker strictly on train split
    logger.info("Fitting Bayesian LaneMatchupTracker on train split (%d games)...", len(splits["train"]))
    matchup_tracker = LaneMatchupTracker(matchup_weight=5.0)
    for r in splits["train"]:
        fj = r["feature_json"]
        w = r["labels_json"]["winner"]
        matchup_tracker.update_match(fj.get("team_a") or [], fj.get("team_b") or [], winner=w)

    pipeline = DraftFeaturePipeline(matchup_tracker=matchup_tracker)

    # 2. Extract 14-dim player tensors
    logger.info("Extracting 14-dim player tensors for all splits...")
    x = {}
    for name, rows in splits.items():
        start_t = time.monotonic()
        tensors_dict = pipeline.batch_extract_tensors([r["feature_json"] for r in rows])
        x[name] = tensors_dict
        logger.info("  Extracted %s tensors (%d games) in %.2fs", name, len(rows), time.monotonic() - start_t)

    y = {name: np.asarray([r["labels_json"]["winner"] for r in rows], dtype=int)
         for name, rows in splits.items()}
    train_rate = float(y["train"].mean())

    targets = _build_targets(splits["train"])

    # 3. Build Equivariant DeepSets Model with 14-dim player tensors and Cross-Lane Attention
    logger.info("Building Equivariant DeepSets model (dim=14, use_cross_attention=True)...")
    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_TENSOR_DIM,
        use_cross_attention=True,
        learning_rate=learning_rate,
    )

    loss_fns = {k: tf.keras.losses.get(LOSS_PER_HEAD[k]) for k in LOSS_PER_HEAD}

    # 4. Train with PCGrad multi-task gradient deconfliction
    best_val_loss = float("inf")
    best_epoch = 1
    history = []
    n_train = len(x["train"]["team_a"])

    logger.info("Starting training for %d epochs (batch_size=%d, n_train=%d)...", epochs, batch_size, n_train)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        seen = 0
        t0 = time.monotonic()

        # Shuffle training indices
        perm = np.random.permutation(n_train)
        shuffled_xa = x["train"]["team_a"][perm]
        shuffled_xb = x["train"]["team_b"][perm]
        shuffled_targets = {k: v[perm] for k, v in targets.items()}

        for start in range(0, n_train, batch_size):
            xb = {
                "team_a": shuffled_xa[start:start + batch_size],
                "team_b": shuffled_xb[start:start + batch_size],
            }
            yb = {k: shuffled_targets[k][start:start + batch_size] for k in shuffled_targets if k in model.output_names}
            losses = pcgrad_step(model, model.optimizer, xb, yb, loss_fns, loss_weights=LOSS_WEIGHTS)
            batch_loss = sum(losses.values())
            total_loss += batch_loss * len(xb["team_a"])
            seen += len(xb["team_a"])

        epoch_time = time.monotonic() - t0
        avg_train_loss = total_loss / max(1, seen)

        # Validation evaluation
        val_preds = model.predict(x["validation"], batch_size=128, verbose=0)
        val_probs = np.asarray(val_preds["winner"])[:, 1]
        val_eval = evaluate_winner(y["validation"], val_probs, train_rate)
        val_log_loss = val_eval["model"]["log_loss"]
        val_acc = val_eval["model"]["accuracy"]

        logger.info(
            "Epoch %d/%d (%.1fs): train_loss=%.4f, val_winner_loss=%.4f, val_acc=%.4f",
            epoch, epochs, epoch_time, avg_train_loss, val_log_loss, val_acc
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_log_loss": val_log_loss,
            "val_accuracy": val_acc,
        })

        if val_log_loss < best_val_loss:
            best_val_loss = val_log_loss
            best_epoch = epoch
            logger.info("  * New best validation log loss (%.4f) -> saving checkpoint to %s", val_log_loss, checkpoint_path)
            model.save(checkpoint_path)

    # 5. Reload best checkpoint for final evaluation
    logger.info("Reloading best checkpoint from epoch %d (val_loss=%.4f)...", best_epoch, best_val_loss)
    from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
    from core.equivariant_model import ThreeClassEquivariantSoftmax
    custom_objects = {
        "SiameseTeamEncoder": SiameseTeamEncoder,
        "AntiSymmetricSoftmax": AntiSymmetricSoftmax,
        "ThreeClassEquivariantSoftmax": ThreeClassEquivariantSoftmax,
    }
    best_model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)

    # 6. Fit Temperature Scaling on Validation Split
    val_preds = best_model.predict(x["validation"], batch_size=128, verbose=0)
    fit_res = fit_temperature(raw_val_probs, y["validation"])
    fitted_temp = float(fit_res["temperature"]) if isinstance(fit_res, dict) else float(fit_res)
    logger.info("Fitted temperature scaling: T = %.4f", fitted_temp)

    calib_artifact = {
        "temperature": fitted_temp,
        "fitted_on": "validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostics": fit_res if isinstance(fit_res, dict) else {},
    }
    with open(out_path / "model.calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_artifact, f, indent=2)

    # 7. Evaluate on untouched Test Split
    test_preds = best_model.predict(x["test"], batch_size=128, verbose=0)
    raw_test_probs = np.asarray(test_preds["winner"])[:, 1]
    cal_test_probs = apply_temperature_to_probabilities(raw_test_probs, fitted_temp)

    test_eval_raw = evaluate_winner(y["test"], raw_test_probs, train_rate, diagnostics=True)
    test_eval_cal = evaluate_winner(y["test"], cal_test_probs, train_rate, diagnostics=True)

    # Selective classification tables
    stratification = {}
    for margin in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
        mask = np.abs(cal_test_probs - 0.5) >= margin
        if mask.any():
            acc = float(np.mean((cal_test_probs[mask] >= 0.5) == y["test"][mask]))
            stratification[f"conviction_ge_{int(margin*100)}pct"] = {
                "coverage_pct": round(float(mask.mean() * 100), 1),
                "count": int(mask.sum()),
                "accuracy": round(acc, 4),
            }

    # Save manifest referencing match IDs
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {
            name: [{"match_id": r["match_id"], "game_creation_time": r["game_creation_time"]}
                   for r in splits[name]]
            for name in ("train", "validation", "test")
        }
    }
    with open(out_path / "model.split.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    metrics_report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "feature_version": "deepsets_v2",
        "player_feature_dim": ENRICHED_PLAYER_TENSOR_DIM,
        "use_cross_attention": True,
        "selected_epoch": best_epoch,
        "temperature": fitted_temp,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "test_metrics_raw": {
            "accuracy": test_eval_raw["model"]["accuracy"],
            "accuracy_lift": test_eval_raw["accuracy_lift"],
            "roc_auc": test_eval_raw["model"]["roc_auc"],
            "log_loss": test_eval_raw["model"]["log_loss"],
            "brier": test_eval_raw["model"]["brier"],
        },
        "test_metrics_calibrated": {
            "accuracy": test_eval_cal["model"]["accuracy"],
            "accuracy_lift": test_eval_cal["accuracy_lift"],
            "roc_auc": test_eval_cal["model"]["roc_auc"],
            "log_loss": test_eval_cal["model"]["log_loss"],
            "brier": test_eval_cal["model"]["brier"],
            "calibration_error": test_eval_cal["model"]["calibration_error"],
        },
        "confidence_stratification": stratification,
        "baseline_accuracy": test_eval_raw["baseline"]["accuracy"],
        "history": history,
    }

    with open(out_path / "model.metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    logger.info("Training and evaluation complete. Results:\n%s", json.dumps(metrics_report, indent=2))
    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-cache", default="runs/candidate_selective_prod/splits_cache_45000.pkl")
    parser.add_argument("--output-dir", default="runs/candidate_deepsets_v2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_deepsets_v2(
        splits_cache=args.splits_cache,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
