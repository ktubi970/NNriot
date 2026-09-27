"""Train Equivariant DeepSets v3 with 14-dim player tensors, macro clash features, and no cross-attention."""
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
from core.features.draft_composition import MACRO_CLASH_FEATURE_DIM
from core.equivariant_model import build_equivariant_deepsets_model, ThreeClassEquivariantSoftmax
from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
from core.generate_graph import LOSS_WEIGHTS, LOSS_PER_HEAD
from core.training_utils import _build_targets
from core.pcgrad import pcgrad_step
from core.training_metrics import evaluate_winner
from core.calibration import fit_temperature, apply_temperature_to_probabilities, save_calibration_artifact
from core.experiment import read_manifest, restore_splits
from core.point_in_time_dataset import load_enriched_records
from core.selective_classification import evaluate_selective_coverage, DEFAULT_CONVICTION_THRESHOLDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_deepsets_v3")


def train_deepsets_v3(
    db_path: str,
    output_dir: str,
    manifest_path: str | None = None,
    splits_cache: str | None = None,
    samples: int = 45000,
    warmup: int = 3000,
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

    # 1. Load or build splits
    splits = None
    if splits_cache:
        cache_file = Path(splits_cache).resolve()
        if cache_file.exists():
            logger.info("Loading cached splits from %s...", cache_file)
            with open(cache_file, "rb") as f:
                splits = pickle.load(f)
            logger.info("Loaded splits: %s", {k: len(v) for k, v in splits.items()})

    if splits is None:
        logger.info("Streaming %d enriched records (warmup=%d) from %s...", samples, warmup, db_path)
        records = load_enriched_records(db_path=db_path, limit=samples, warmup_matches=warmup)
        
        if manifest_path and Path(manifest_path).exists():
            logger.info("Restoring exact splits from manifest %s...", manifest_path)
            manifest = read_manifest(Path(manifest_path))
            splits = restore_splits(records, manifest)
        else:
            logger.info("No manifest provided, creating chronological 70/15/15 splits...")
            from core.pre_match_dataset import split_records
            splits = split_records(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

        if splits_cache:
            cache_file = Path(splits_cache).resolve()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Caching splits to %s...", cache_file)
            with open(cache_file, "wb") as f:
                pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Using splits: %s", {k: len(v) for k, v in splits.items()})

    # 2. Build Bayesian smoothed LaneMatchupTracker strictly on train split
    logger.info("Fitting Bayesian LaneMatchupTracker on train split (%d games)...", len(splits["train"]))
    matchup_tracker = LaneMatchupTracker(matchup_weight=5.0)
    for r in splits["train"]:
        fj = r["feature_json"]
        w = r["labels_json"]["winner"]
        matchup_tracker.update_match(fj.get("team_a") or [], fj.get("team_b") or [], winner=w)

    pipeline = DraftFeaturePipeline(matchup_tracker=matchup_tracker)

    # 3. Extract 14-dim player tensors and 15-dim macro clash vectors using batch_extract_v3
    logger.info("Extracting v3 features (14-dim player tensors + 15-dim macro clash) for all splits...")
    x = {}
    for name, rows in splits.items():
        start_t = time.monotonic()
        feat_dicts = [r["feature_json"] for r in rows]
        x[name] = pipeline.batch_extract_v3(feat_dicts)
        logger.info(
            "  Extracted %s: team_a %s, macro_clash %s (%d games) in %.2fs",
            name,
            x[name]["team_a"].shape,
            x[name]["macro_clash"].shape,
            len(rows),
            time.monotonic() - start_t,
        )

    y = {name: np.asarray([r["labels_json"]["winner"] for r in rows], dtype=int)
         for name, rows in splits.items()}
    train_rate = float(y["train"].mean())

    targets = _build_targets(splits["train"])

    # 4. Build Equivariant DeepSets Model with 14-dim tensors and 15-dim macro clash (no cross-attention)
    logger.info(
        "Building Equivariant DeepSets model (dim=%d, macro_dim=%d, use_cross_attention=False)...",
        ENRICHED_PLAYER_TENSOR_DIM,
        MACRO_CLASH_FEATURE_DIM,
    )
    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_TENSOR_DIM,
        include_macro_clash=True,
        macro_feature_dim=MACRO_CLASH_FEATURE_DIM,
        use_cross_attention=False,
        learning_rate=learning_rate,
    )

    loss_fns = {k: tf.keras.losses.get(LOSS_PER_HEAD[k]) for k in LOSS_PER_HEAD}

    # 5. Train with PCGrad multi-task gradient deconfliction
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
        shuffled_targets = {k: v[perm] for k, v in targets.items()}

        for start in range(0, n_train, batch_size):
            xb = {
                "team_a": shuffled_xa[start:start + batch_size],
                "team_b": shuffled_xb[start:start + batch_size],
                "macro_clash": shuffled_xm[start:start + batch_size],
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

    # 6. Reload best checkpoint for final evaluation
    logger.info("Reloading best checkpoint from epoch %d (val_loss=%.4f)...", best_epoch, best_val_loss)
    custom_objects = {
        "SiameseTeamEncoder": SiameseTeamEncoder,
        "AntiSymmetricSoftmax": AntiSymmetricSoftmax,
        "ThreeClassEquivariantSoftmax": ThreeClassEquivariantSoftmax,
    }
    best_model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)

    # 7. Fit Temperature Scaling on Validation Split
    val_preds = best_model.predict(x["validation"], batch_size=128, verbose=0)
    raw_val_probs = np.asarray(val_preds["winner"])[:, 1]
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

    # 8. Evaluate on untouched Test Split (6,750 games)
    test_preds = best_model.predict(x["test"], batch_size=128, verbose=0)
    raw_test_probs = np.asarray(test_preds["winner"])[:, 1]
    cal_test_probs = apply_temperature_to_probabilities(raw_test_probs, fitted_temp)

    test_eval_raw = evaluate_winner(y["test"], raw_test_probs, train_rate, diagnostics=True)
    test_eval_cal = evaluate_winner(y["test"], cal_test_probs, train_rate, diagnostics=True)

    # 9. Selective classification tables with Wilson 95% CIs
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

    # Save comprehensive metrics
    metrics = {
        "model_name": "EquivariantDeepSets_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "player_feature_dim": ENRICHED_PLAYER_TENSOR_DIM,
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
    with open(out_path / "model.metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=== DeepSets v3 Training Complete ===")
    logger.info("Unconditional Test Accuracy: %.2f%% (log-loss: %.4f)",
                test_eval_cal["model"]["accuracy"] * 100, test_eval_cal["model"]["log_loss"])
    for k, v in stratification.items():
        logger.info("  %s: Coverage=%s%%, Acc=%.2f%%, CI=[%.2f%%, %.2f%%]",
                    k, v["coverage_pct"], v["accuracy"] * 100,
                    v["wilson_ci_95"][0] * 100, v["wilson_ci_95"][1] * 100)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="runs/db_merge_20260924/consolidated.db")
    parser.add_argument("--output-dir", default="runs/candidate_deepsets_v3")
    parser.add_argument("--manifest-path", default="runs/candidate_selective_prod/model.split.json")
    parser.add_argument("--splits-cache", default="runs/candidate_deepsets_v3/splits_cache_45000_v3.pkl")
    parser.add_argument("--samples", type=int, default=45000)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_deepsets_v3(
        db_path=args.db_path,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        splits_cache=args.splits_cache,
        samples=args.samples,
        warmup=args.warmup,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
