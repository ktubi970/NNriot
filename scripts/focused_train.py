#!/usr/bin/env python3
"""
focused_train.py - Diagnostic one-shot training to localize NN performance bug.

Loads N samples, builds a FRESH multi-output model (no checkpoint), trains
with a clean 80/20 split for M epochs, logs val winner accuracy each epoch.

Interpretation:
  - val_winner_acc climbs to 0.8+ -> continuous training loop is broken
  - val_winner_acc stays near baseline -> architecture/regularization is the bug
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import logging
import time

import numpy as np
import tensorflow as tf

from core import database
from core import json_utils
from scripts.continuous_trainer import _build_targets, sparse_microbatch_generator
from core.feature_labels import LABEL_KEYS, VECTOR_DIM
from generate_graph import build_multi_output_model

N_SAMPLES = 5000
TEST_FRACTION = 0.2
N_EPOCHS = 20
BATCH = 64
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("focused_train")


def main() -> None:
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    logger.info("Fetching up to %d records ...", N_SAMPLES)
    rows = database.get_random_trained_records(limit=N_SAMPLES)
    rows = [r for r in rows if r.get("feature_json") and r.get("labels_json")]
    # Match continuous_trainer's filter so we don't crash on incomplete labels.
    required = set(LABEL_KEYS)
    rows = [r for r in rows if required.issubset(r["labels_json"].keys())]
    logger.info("Got %d usable records.", len(rows))
    if len(rows) < 1000:
        raise SystemExit("Need at least 1000 records for a meaningful experiment.")

    # ── Split ────────────────────────────────────────────────────────────
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    n_val = int(len(rows) * TEST_FRACTION)
    val_idx = set(indices[:n_val].tolist())
    train_records = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val_records = [rows[i] for i in range(len(rows)) if i in val_idx]
    logger.info("Train: %d  Val: %d", len(train_records), len(val_records))

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
    # Convert val to dense once (it's small: ~1000 * 20k * 4 = ~80MB).
    x_val_dense = x_val_sp.toarray()
    logger.info("Vectorized + targeted in %.2fs.", time.time() - t0)

    # ── Build fresh model ────────────────────────────────────────────────
    logger.info("Building fresh model (no checkpoint) ...")
    from core.model_factory import ModelFactory
    model, _ = ModelFactory.get_model(version="v2", input_dim=VECTOR_DIM, force_rebuild=True)
    logger.info(
        "Model params: %s   heads: %d",
        f"{model.count_params():,}", len(model.output_names),
    )

    # ── Train ────────────────────────────────────────────────────────────
    history: list[dict] = []
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        batch_losses: list[float] = []
        for x_batch, y_batch in sparse_microbatch_generator(x_train_sp, y_train, batch_size=BATCH):
            losses = model.train_on_batch(x_batch, y_batch, return_dict=True)
            batch_losses.append(float(losses["loss"]))
        train_loss = float(np.mean(batch_losses))

        # Validation: evaluate in chunks to bound peak memory.
        val_metrics = model.evaluate(x_val_dense, y_val, batch_size=BATCH, verbose=0, return_dict=True)
        # Names look like "winner_accuracy", "total_kills_mae", etc.
        win_acc = float(val_metrics.get("winner_accuracy", 0.0))
        kills_mae = float(val_metrics.get("total_kills_mae", 0.0))
        elder_acc = float(val_metrics.get("elder_dragon_accuracy", 0.0))
        fb_acc = float(val_metrics.get("first_blood_accuracy", 0.0))
        val_loss = float(val_metrics.get("loss", 0.0))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_winner_acc": win_acc,
            "val_first_blood_acc": fb_acc,
            "val_elder_dragon_acc": elder_acc,
            "val_total_kills_mae": kills_mae,
            "duration_s": round(time.time() - t0, 2),
        })
        logger.info(
            "epoch %2d/%d  train_loss=%.3f  val_loss=%.3f  val_winner_acc=%.3f  "
            "val_first_blood_acc=%.3f  val_total_kills_mae=%.2f  [%.1fs]",
            epoch, N_EPOCHS, train_loss, val_loss, win_acc, fb_acc, kills_mae,
            time.time() - t0,
        )

    # ── Baseline reference ───────────────────────────────────────────────
    y_val_winner = np.argmax(y_val["winner"], axis=1)
    baseline = float(max(y_val_winner.mean(), 1 - y_val_winner.mean()))
    best = max(h["val_winner_acc"] for h in history)
    final = history[-1]["val_winner_acc"]

    print()
    print("=== Focused training summary ===")
    print(f"Records: train={len(train_records)}  val={len(val_records)}")
    print(f"Epochs: {N_EPOCHS}   batch: {BATCH}")
    print(f"Best val winner acc: {best:.4f}")
    print(f"Final val winner acc: {final:.4f}")
    print(f"Majority baseline: {baseline:.4f}")
    print(f"Lift over baseline (best): {best - baseline:+.4f}")
    print()
    if best - baseline > 0.10:
        print("VERDICT: NN architecture CAN learn from these features.")
        print("         The continuous training loop is the bug (under-iteration,")
        print("         experience-replay imbalance, or checkpoint thrashing).")
    elif best - baseline > 0.02:
        print("VERDICT: NN learns weakly. Architecture or regularization may be")
        print("         drowning the signal that logreg captured.")
    else:
        print("VERDICT: NN does NOT learn even on a clean one-shot split.")
        print("         Architecture/dropout/loss-weighting is the bug.")

    out_path = f"focused_train_history_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_train": len(train_records),
            "n_val": len(val_records),
            "n_epochs": N_EPOCHS,
            "batch": BATCH,
            "baseline_winner": baseline,
            "history": history,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
