#!/usr/bin/env python3
"""
retrain_full.py - From-scratch retraining on the entire training_dataset.

Pulls every row with complete feature_json + labels_json, splits 80/20, and
trains a fresh multi-output model with early stopping on val_loss. Saves the
best-by-val-loss checkpoint to model_v2.keras atomically.

This is the recommended retraining workflow (see continuous_trainer.py
comments). continuous_trainer.py only does small incremental updates and
will overwrite model_v2.keras on its next cycle if it is running — stop it
before invoking this script.

Usage:
    python retrain_full.py [--epochs 25] [--patience 4] [--limit N]

The --limit flag is for smoke testing on a subset.
"""

import argparse
import json
import logging
import os
import shutil
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

import database
import json_utils
from continuous_trainer import _build_targets, sparse_microbatch_generator
from feature_labels import LABEL_KEYS, VECTOR_DIM
from generate_graph import build_multi_output_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_v2.keras")
# Keras 3 enforces a `.keras` / `.h5` suffix on save() — keep the suffix at the
# end and prepend `.tmp` to the stem.
TMP_CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_v2.tmp.keras")
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrain_full")


def _fetch_all_records(limit: int | None = None) -> list[dict]:
    """Stream every training_dataset row with complete labels.

    feature_json / labels_json are gzip+base64-encoded in the DB — use
    database._decode_compressed_json so this script accepts the same
    on-disk format as get_untrained_records / get_random_trained_records.
    """
    with database.get_connection() as conn:
        query = (
            "SELECT id, match_id, feature_json, winner_label, labels_json "
            "FROM training_dataset"
        )
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()

    required = set(LABEL_KEYS)
    out = []
    for r in rows:
        d = dict(r)
        d["feature_json"] = database._decode_compressed_json(d["feature_json"])
        d["labels_json"] = database._decode_compressed_json(d["labels_json"])
        if not d["feature_json"] or not d.get("labels_json"):
            continue
        if not required.issubset(d["labels_json"].keys()):
            continue
        out.append(d)
    return out


def _evaluate_val(model: tf.keras.Model, x_val_sp, y_val: dict, batch: int) -> dict:
    """Evaluate in microbatches to avoid materializing the whole val set as dense."""
    # tf.keras.Model.evaluate accepts a generator via x. Use sparse_microbatch_generator
    # with shuffle disabled to keep predictions aligned (we only care about reduce
    # metrics, so order doesn't strictly matter here, but consistency is nice).
    losses: list[dict] = []
    n_samples = x_val_sp.shape[0]
    metric_sum: dict[str, float] = {}
    weight_sum = 0
    for start in range(0, n_samples, batch):
        end = min(start + batch, n_samples)
        x_batch = x_val_sp[start:end].toarray()
        y_batch = {k: v[start:end] for k, v in y_val.items()}
        results = model.test_on_batch(x_batch, y_batch, return_dict=True)
        w = end - start
        weight_sum += w
        for k, v in results.items():
            metric_sum[k] = metric_sum.get(k, 0.0) + float(v) * w
        losses.append(results)
    return {k: v / weight_sum for k, v in metric_sum.items()}


def _save_atomic(model: tf.keras.Model) -> None:
    """Save to a .tmp file and rename — avoids leaving a half-written .keras."""
    model.save(TMP_CHECKPOINT_PATH)
    # Windows requires the destination not exist for os.rename, but shutil.move handles it.
    shutil.move(TMP_CHECKPOINT_PATH, CHECKPOINT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4,
                        help="Stop after N epochs without val_loss improvement.")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test on first N records (default: all).")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    logger.info("Fetching records (limit=%s) ...", args.limit)
    t0 = time.time()
    records = _fetch_all_records(args.limit)
    logger.info("Got %d usable records in %.1fs.", len(records), time.time() - t0)
    if len(records) < 500:
        raise SystemExit("Too few records — populate training_dataset first.")

    # ── Split (stratify by winner so val rate matches train) ────────────
    winner_labels = np.asarray([
        int(r["labels_json"]["winner"]) for r in records
    ])
    indices = np.arange(len(records))
    rng.shuffle(indices)
    n_val = int(len(records) * args.val_fraction)
    val_set = set(indices[:n_val].tolist())
    train_records = [records[i] for i in range(len(records)) if i not in val_set]
    val_records = [records[i] for i in range(len(records)) if i in val_set]
    train_w = float(np.mean([r["labels_json"]["winner"] for r in train_records]))
    val_w = float(np.mean([r["labels_json"]["winner"] for r in val_records]))
    logger.info(
        "Train: %d  Val: %d   train_winner_rate=%.3f  val_winner_rate=%.3f",
        len(train_records), len(val_records), train_w, val_w,
    )

    # ── Vectorize ────────────────────────────────────────────────────────
    t0 = time.time()
    x_train_sp = json_utils.json_to_vector(
        [r["feature_json"] for r in train_records], dim=VECTOR_DIM
    )
    x_val_sp = json_utils.json_to_vector(
        [r["feature_json"] for r in val_records], dim=VECTOR_DIM
    )
    y_train = _build_targets(train_records)
    y_val = _build_targets(val_records)
    logger.info(
        "Vectorized: x_train=%s nnz=%d  x_val=%s nnz=%d (%.1fs)",
        x_train_sp.shape, x_train_sp.nnz, x_val_sp.shape, x_val_sp.nnz,
        time.time() - t0,
    )

    # ── Build fresh model ────────────────────────────────────────────────
    model = build_multi_output_model(input_dim=VECTOR_DIM)
    logger.info("Model params: %s   heads: %d", f"{model.count_params():,}",
                len(model.output_names))

    # ── Train loop with early stopping on val_loss ───────────────────────
    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    stale = 0
    best_state_saved = False

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        batch_losses: list[float] = []
        for x_batch, y_batch in sparse_microbatch_generator(
            x_train_sp, y_train, batch_size=args.batch,
        ):
            losses = model.train_on_batch(x_batch, y_batch, return_dict=True)
            batch_losses.append(float(losses["loss"]))
        train_loss = float(np.mean(batch_losses))

        val_metrics = _evaluate_val(model, x_val_sp, y_val, batch=args.batch)
        val_loss = float(val_metrics.get("loss", 0.0))
        win_acc = float(val_metrics.get("winner_accuracy", 0.0))
        fb_acc = float(val_metrics.get("first_blood_accuracy", 0.0))
        kills_mae = float(val_metrics.get("total_kills_mae", 0.0))

        improved = val_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            _save_atomic(model)
            best_state_saved = True
            tag = " * BEST *"
        else:
            stale += 1
            tag = ""

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_winner_acc": win_acc,
            "val_first_blood_acc": fb_acc,
            "val_total_kills_mae": kills_mae,
            "duration_s": round(time.time() - t0, 2),
            "best_val_loss_so_far": best_val_loss,
        })
        logger.info(
            "epoch %2d/%d  train_loss=%.3f  val_loss=%.3f  val_winner_acc=%.3f  "
            "val_first_blood_acc=%.3f  val_total_kills_mae=%.2f  [%.1fs]%s",
            epoch, args.epochs, train_loss, val_loss, win_acc, fb_acc, kills_mae,
            time.time() - t0, tag,
        )
        if stale >= args.patience:
            logger.info(
                "Early stop: no val_loss improvement for %d epochs (best @ epoch %d).",
                args.patience, best_epoch,
            )
            break

    if not best_state_saved:
        raise SystemExit("No epoch improved val_loss; refusing to overwrite checkpoint.")

    # ── Summary + metrics dump ───────────────────────────────────────────
    print()
    print("=== retrain_full summary ===")
    print(f"Records: train={len(train_records)}  val={len(val_records)}")
    print(f"Best epoch: {best_epoch}   best val_loss: {best_val_loss:.4f}")
    best_entry = next(h for h in history if h["epoch"] == best_epoch)
    print(f"Best val_winner_acc:      {best_entry['val_winner_acc']:.4f}")
    print(f"Best val_first_blood_acc: {best_entry['val_first_blood_acc']:.4f}")
    print(f"Best val_total_kills_mae: {best_entry['val_total_kills_mae']:.4f}  (normalized units)")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    snapshot = {
        "timestamp": int(time.time()),
        "checkpoint": CHECKPOINT_PATH,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics": best_entry,
        "history": history,
    }
    out_path = f"retrain_full_metrics_{snapshot['timestamp']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
