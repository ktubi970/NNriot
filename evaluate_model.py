#!/usr/bin/env python3
"""
evaluate_model.py - One-off held-out evaluation of model_v2.keras.

Loads the current checkpoint, samples N records from the training_dataset,
and reports per-head accuracy / MAE. Writes a JSON snapshot so we can
diff against future runs.

Usage:
    python evaluate_model.py [--samples 1000]

Notes
-----
- We sample from is_trained=1 records (the trainer marks rows after training
  on them, so this is "training set" not a true holdout — see the report
  for caveats). When is_trained=0 records exist they are used as a true
  holdout instead.
- Regression MAE is reported in **raw units** (kills, towers, etc.), i.e.
  after un-normalizing the model output via REGRESSION_STATS.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import datetime as _dt
import json
import logging

import numpy as np
import tensorflow as tf

import database
import json_utils
from continuous_trainer import REGRESSION_STATS
from feature_labels import ALL_LABEL_KEYS, VECTOR_DIM

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_v2.keras")

# Head taxonomy (mirrors continuous_trainer._build_targets shapes)
SOFTMAX_2_HEADS = ("winner", "team_b_kill_lead")
SOFTMAX_3_HEADS = ("first_blood", "first_baron", "first_inhibitor", "first_tower",
                   "first_to_5_kills", "first_to_10_kills",
                   "first_to_15_kills", "first_to_20_kills")
BINARY_HEADS = ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon")
REGRESSION_HEADS = tuple(REGRESSION_STATS.keys())

# Default class for first_to_N when missing (mirrors trainer behavior)
FIRST_TO_DEFAULT = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_model")


def _fetch_records(n: int) -> list[dict]:
    """Prefer untrained holdout rows; fall back to a random sample of trained rows."""
    holdout = [r for r in database.get_untrained_records(limit=n)
               if r.get("feature_json") and r.get("labels_json")]
    if len(holdout) >= max(50, n // 4):
        logger.info("Using %d untrained rows as a true holdout.", len(holdout))
        return holdout
    sample = database.get_random_trained_records(limit=n)
    logger.info(
        "Only %d complete untrained rows; sampling %d trained rows instead (not a true holdout).",
        len(holdout), len(sample),
    )
    return sample


def _label_for_head(head: str, labels: dict) -> float | int | None:
    """Pull the ground-truth value for a single head from a labels_json dict."""
    if head in REGRESSION_HEADS:
        if head not in labels:
            return None
        return float(labels[head])
    if head.startswith("first_to_"):
        return int(labels.get(head, FIRST_TO_DEFAULT))
    if head not in labels:
        return None
    return int(labels[head])


def evaluate(n_samples: int) -> dict:
    logger.info("Loading model from %s ...", CHECKPOINT_PATH)
    model = tf.keras.models.load_model(CHECKPOINT_PATH)
    logger.info("Model loaded. Output heads: %d", len(model.output_names))

    logger.info("Fetching %d records ...", n_samples)
    records = _fetch_records(n_samples)
    records = [r for r in records if r.get("feature_json") and r.get("labels_json")]
    if not records:
        raise SystemExit("No records with feature_json + labels_json available.")
    logger.info("Evaluating on %d records.", len(records))

    json_features = [r["feature_json"] for r in records]
    x_sparse = json_utils.json_to_vector(json_features, dim=VECTOR_DIM)

    # Predict in batches to keep memory bounded (dense conversion is 80MB/1000 rows).
    BATCH = 200
    preds: dict[str, list[np.ndarray]] = {h: [] for h in ALL_LABEL_KEYS}
    for start in range(0, x_sparse.shape[0], BATCH):
        x_batch = x_sparse[start:start + BATCH].toarray()
        out = model.predict(x_batch, verbose=0)
        # Keras may return a dict, a list, or a single ndarray depending on
        # how the model was compiled. Normalize to {head_name: ndarray}.
        if isinstance(out, dict):
            for name, arr in out.items():
                preds[name].append(np.asarray(arr))
        elif isinstance(out, list):
            for name, arr in zip(model.output_names, out):
                preds[name].append(np.asarray(arr))
        else:
            preds[model.output_names[0]].append(np.asarray(out))
    preds_arr: dict[str, np.ndarray] = {k: np.concatenate(v, axis=0) for k, v in preds.items() if v}

    # ── Per-head metrics ──────────────────────────────────────────────────
    results: dict[str, dict] = {}

    for head in SOFTMAX_2_HEADS + SOFTMAX_3_HEADS:
        y_true = []
        y_pred_class = []
        for i, r in enumerate(records):
            true = _label_for_head(head, r["labels_json"])
            if true is None:
                continue
            y_true.append(true)
            y_pred_class.append(int(np.argmax(preds_arr[head][i])))
        if not y_true:
            results[head] = {"type": "classification", "n": 0}
            continue
        y_true_a = np.asarray(y_true)
        y_pred_a = np.asarray(y_pred_class)
        acc = float((y_true_a == y_pred_a).mean())
        baseline = float(np.bincount(y_true_a).max() / len(y_true_a))
        results[head] = {
            "type": "classification",
            "n": len(y_true),
            "accuracy": round(acc, 4),
            "majority_baseline": round(baseline, 4),
            "lift_over_baseline": round(acc - baseline, 4),
        }

    for head in BINARY_HEADS:
        y_true = []
        y_pred_class = []
        for i, r in enumerate(records):
            true = _label_for_head(head, r["labels_json"])
            if true is None:
                continue
            y_true.append(true)
            y_pred_class.append(int(preds_arr[head][i].flatten()[0] > 0.5))
        if not y_true:
            results[head] = {"type": "binary", "n": 0}
            continue
        y_true_a = np.asarray(y_true)
        y_pred_a = np.asarray(y_pred_class)
        acc = float((y_true_a == y_pred_a).mean())
        positive_rate = float(y_true_a.mean())
        baseline = max(positive_rate, 1 - positive_rate)
        results[head] = {
            "type": "binary",
            "n": len(y_true),
            "accuracy": round(acc, 4),
            "positive_rate": round(positive_rate, 4),
            "majority_baseline": round(baseline, 4),
            "lift_over_baseline": round(acc - baseline, 4),
        }

    for head in REGRESSION_HEADS:
        mean, std = REGRESSION_STATS[head]
        y_true_raw = []
        y_pred_raw = []
        for i, r in enumerate(records):
            true = _label_for_head(head, r["labels_json"])
            if true is None:
                continue
            # Model output is normalized; convert back to raw units.
            pred_norm = float(preds_arr[head][i].flatten()[0])
            y_true_raw.append(true)
            y_pred_raw.append(pred_norm * std + mean)
        if not y_true_raw:
            results[head] = {"type": "regression", "n": 0}
            continue
        y_true_a = np.asarray(y_true_raw, dtype=np.float64)
        y_pred_a = np.asarray(y_pred_raw, dtype=np.float64)
        mae = float(np.mean(np.abs(y_true_a - y_pred_a)))
        # Baseline: predict the mean of y_true.
        baseline_mae = float(np.mean(np.abs(y_true_a - y_true_a.mean())))
        results[head] = {
            "type": "regression",
            "n": len(y_true_raw),
            "mae": round(mae, 3),
            "mean_baseline_mae": round(baseline_mae, 3),
            "lift_over_baseline": round(baseline_mae - mae, 3),
            "y_true_mean": round(float(y_true_a.mean()), 3),
            "y_true_std": round(float(y_true_a.std()), 3),
        }

    snapshot = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_size_bytes": os.path.getsize(CHECKPOINT_PATH),
        "n_records": len(records),
        "results": results,
    }
    return snapshot


def _print_table(snapshot: dict) -> None:
    print()
    print(f"=== Model evaluation @ {snapshot['timestamp']} ===")
    print(f"Records evaluated: {snapshot['n_records']}")
    print()
    header = f"{'head':24s} {'type':12s} {'n':>5s}  {'metric':>10s}  {'baseline':>10s}  {'lift':>8s}"
    print(header)
    print("-" * len(header))
    for head, r in snapshot["results"].items():
        if r.get("n", 0) == 0:
            print(f"{head:24s} {r['type']:12s} {0:>5d}   (no labeled samples)")
            continue
        if r["type"] == "regression":
            metric = f"{r['mae']:>10.3f}"
            base = f"{r['mean_baseline_mae']:>10.3f}"
            lift = f"{r['lift_over_baseline']:>+8.3f}"
        else:
            metric = f"{r['accuracy']:>10.3f}"
            base = f"{r['majority_baseline']:>10.3f}"
            lift = f"{r['lift_over_baseline']:>+8.3f}"
        print(f"{head:24s} {r['type']:12s} {r['n']:>5d}  {metric}  {base}  {lift}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to JSON snapshot (default: metrics_<timestamp>.json)")
    args = parser.parse_args()

    snapshot = evaluate(args.samples)
    _print_table(snapshot)

    out_path = args.output or f"metrics_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
