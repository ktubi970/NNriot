"""Train Equivariant DeepSets v4 with 18-dim player tensors, macro clash features, and timeline priors."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.features.pipeline import DraftFeaturePipeline, ENRICHED_PLAYER_TIMELINE_TENSOR_DIM
from core.features.draft_composition import MACRO_CLASH_FEATURE_DIM
from core.equivariant_model import build_equivariant_deepsets_model, NNRIOT_CUSTOM_OBJECTS
from core.generate_graph import LOSS_WEIGHTS, LOSS_PER_HEAD
from core.training_utils import _build_targets
from core.pcgrad import pcgrad_step
from core.training_metrics import evaluate_winner
from core.calibration import (
    fit_temperature,
    apply_temperature_to_probabilities,
    save_calibration_artifact,
)
from core.experiment import build_manifest, read_manifest, restore_splits
from core.point_in_time_dataset import load_enriched_records
from core.pre_match_dataset import split_records
from core.selective_classification import evaluate_selective_coverage, DEFAULT_CONVICTION_THRESHOLDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_deepsets_v4_timeline")


def train_deepsets_v4_timeline(
    db_path: str,
    output_dir: str,
    samples: int = 45000,
    warmup: int = 3000,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
    learning_rate: float = 0.0008,
    *,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    """Train Equivariant DeepSets v4 with timeline player priors and evaluate artifacts."""
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir).resolve()
    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path / "model.keras"
    split_manifest_path = out_path / "model.split.json"

    # 1. Load chronologically streaming enriched records with timeline priors enabled
    logger.info(
        "Loading enriched records with include_timeline=True (limit=%d, warmup=%d) from %s...",
        samples, warmup, db_path,
    )
    records = load_enriched_records(
        db_path=db_path,
        limit=samples,
        warmup_matches=warmup,
        include_timeline=True,
    )
    if not records:
        raise ValueError(f"No valid records loaded from database: {db_path}")

    # 2. Build or restore splits
    if manifest_path:
        logger.info("Restoring exact splits from manifest %s...", manifest_path)
        manifest = read_manifest(Path(manifest_path))
        splits = restore_splits(records, manifest)
    else:
        logger.info("Creating chronological 70/15/15 splits...")
        splits, split_meta = split_records(records, strategy="chronological")
        manifest = build_manifest(splits, source_db=db_path, split_metadata=split_meta)

    with split_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    logger.info("Split record counts: %s", {k: len(v) for k, v in splits.items()})

    # 3. Extract 18-dim player tensors and 15-dim macro clash features
    pipeline = DraftFeaturePipeline()
    x: dict[str, dict[str, np.ndarray]] = {}

    for split_name in ("train", "validation", "test"):
        rows = splits[split_name]
        n_rows = len(rows)
        ta_arr = np.zeros((n_rows, 5, ENRICHED_PLAYER_TIMELINE_TENSOR_DIM), dtype=np.float32)
        tb_arr = np.zeros((n_rows, 5, ENRICHED_PLAYER_TIMELINE_TENSOR_DIM), dtype=np.float32)
        mc_arr = np.zeros((n_rows, MACRO_CLASH_FEATURE_DIM), dtype=np.float32)

        for i, r in enumerate(rows):
            feat = r.get("feature_json") or {}
            ta, tb = pipeline.extract_tensors(feat, include_timeline=True)
            mc = pipeline.extract_macro_clash(feat)
            ta_arr[i] = ta
            tb_arr[i] = tb
            mc_arr[i] = mc

        x[split_name] = {
            "team_a": ta_arr,
            "team_b": tb_arr,
            "macro_clash": mc_arr,
        }

    y = {
        name: np.asarray([r["labels_json"].get("winner", 0) for r in rows], dtype=int)
        for name, rows in splits.items()
    }
    train_rate = float(y["train"].mean()) if len(y["train"]) > 0 else 0.5
    targets = {name: _build_targets(splits[name]) for name in ("train", "validation", "test")}

    # 4. Build Equivariant DeepSets Model with 18-dim player tensors and 15-dim macro clash
    logger.info(
        "Building Equivariant DeepSets model (dim=%d, macro_dim=%d, use_cross_attention=False)...",
        ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        MACRO_CLASH_FEATURE_DIM,
    )
    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        include_macro_clash=True,
        macro_feature_dim=MACRO_CLASH_FEATURE_DIM,
        use_cross_attention=False,
        learning_rate=learning_rate,
    )

    loss_fns = {
        k: tf.keras.losses.get(LOSS_PER_HEAD[k])
        for k in LOSS_PER_HEAD
        if k in model.output_names
    }

    # 5. Multi-task PCGrad training loop
    best_val_loss = float("inf")
    best_epoch = 1
    history = []
    n_train = len(x["train"]["team_a"])

    logger.info("Starting training for %d epochs (batch_size=%d, n_train=%d)...", epochs, batch_size, n_train)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        seen = 0
        t0 = time.monotonic()

        perm = np.random.permutation(n_train)
        shuffled_xa = x["train"]["team_a"][perm]
        shuffled_xb = x["train"]["team_b"][perm]
        shuffled_xm = x["train"]["macro_clash"][perm]
        shuffled_targets = {k: v[perm] for k, v in targets["train"].items() if k in model.output_names}

        for start in range(0, n_train, batch_size):
            xb = {
                "team_a": shuffled_xa[start:start + batch_size],
                "team_b": shuffled_xb[start:start + batch_size],
                "macro_clash": shuffled_xm[start:start + batch_size],
            }
            yb = {k: shuffled_targets[k][start:start + batch_size] for k in shuffled_targets}
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
        val_log_loss = float(val_eval["model"]["log_loss"])
        val_acc = float(val_eval["model"]["accuracy"])

        logger.info(
            "Epoch %d/%d (%.1fs): train_loss=%.4f, val_winner_loss=%.4f, val_acc=%.4f",
            epoch, epochs, epoch_time, avg_train_loss, val_log_loss, val_acc,
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

    # Ensure checkpoint is saved if not already
    if not checkpoint_path.exists():
        model.save(checkpoint_path)

    # 6. Reload best checkpoint for final evaluation
    logger.info("Reloading best checkpoint from %s...", checkpoint_path)
    best_model = tf.keras.models.load_model(checkpoint_path, custom_objects=NNRIOT_CUSTOM_OBJECTS)

    # 7. Temperature calibration on Validation Split
    val_preds = best_model.predict(x["validation"], batch_size=128, verbose=0)
    raw_val_probs = np.asarray(val_preds["winner"])[:, 1]

    fitted_temp = 1.0
    calib_sidecar = checkpoint_path.with_suffix(".calibration.json")
    calib_saved = False

    try:
        if len(y["validation"]) >= 20 and len(np.unique(y["validation"])) == 2:
            fit_res = fit_temperature(raw_val_probs, y["validation"])
            fitted_temp = float(fit_res["temperature"])
            logger.info("Fitted temperature scaling: T = %.4f", fitted_temp)
            save_calibration_artifact(
                checkpoint_path=checkpoint_path,
                split_manifest_path=split_manifest_path,
                fit=fit_res,
                input_dim=ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
            )
            calib_saved = True
        else:
            logger.warning(
                "Skipping strict temperature calibration: validation size (%d) < 20 or single class",
                len(y["validation"]),
            )
    except Exception as exc:
        logger.warning("save_calibration_artifact raised %s; writing fallback artifact.", exc)

    if not calib_saved:
        fallback_artifact = {
            "schema_version": 1,
            "method": "binary_temperature",
            "head": "winner",
            "temperature": fitted_temp,
            "fitted_on": "validation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fallback": True,
        }
        with calib_sidecar.open("w", encoding="utf-8") as f:
            json.dump(fallback_artifact, f, indent=2)

    # 8. Test split evaluation
    test_preds = best_model.predict(x["test"], batch_size=128, verbose=0)
    raw_test_probs = np.asarray(test_preds["winner"])[:, 1]
    cal_test_probs = apply_temperature_to_probabilities(raw_test_probs, fitted_temp)

    test_eval_raw = evaluate_winner(y["test"], raw_test_probs, train_rate, diagnostics=True)
    test_eval_cal = evaluate_winner(y["test"], cal_test_probs, train_rate, diagnostics=True)

    # 9. Selective classification coverage and accuracy
    stratification = {}
    for margin in DEFAULT_CONVICTION_THRESHOLDS:
        sc_res = evaluate_selective_coverage(y["test"], cal_test_probs, threshold=margin)
        stratification[f"conviction_ge_{int(margin*100)}pct"] = {
            "threshold": margin,
            "coverage_pct": round(sc_res["coverage"] * 100, 2),
            "count": sc_res["count"],
            "accuracy": round(sc_res["accuracy"], 4),
            "wilson_ci_95": [round(sc_res["wilson_lower"], 4), round(sc_res["wilson_upper"], 4)],
        }

    # 10. Summary metrics
    metrics = {
        "model_name": "EquivariantDeepSets_v4_timeline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "player_feature_dim": ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        "macro_feature_dim": MACRO_CLASH_FEATURE_DIM,
        "use_cross_attention": False,
        "best_epoch": best_epoch,
        "best_val_log_loss": round(float(best_val_loss), 4),
        "temperature": round(fitted_temp, 4),
        "test_raw": test_eval_raw,
        "test_calibrated": test_eval_cal,
        "selective_classification": stratification,
        "history": history,
    }
    with (out_path / "model.metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=== DeepSets v4 Timeline Training Complete ===")
    logger.info(
        "Test Accuracy: %.2f%% (log-loss: %.4f)",
        test_eval_cal["model"]["accuracy"] * 100,
        test_eval_cal["model"]["log_loss"],
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Equivariant DeepSets v4 with timeline player priors")
    parser.add_argument("--db-path", default="runs/db_merge_20260924/consolidated.db")
    parser.add_argument("--output-dir", default="runs/candidate_deepsets_v4_timeline")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--samples", type=int, default=45000)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_deepsets_v4_timeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        samples=args.samples,
        warmup=args.warmup,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        manifest_path=args.manifest_path,
    )
