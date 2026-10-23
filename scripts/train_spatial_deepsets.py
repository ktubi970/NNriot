"""Train Equivariant DeepSets spatial neural model on matches with trajectory embeddings."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.pre_match_dataset import _decode
from core.database import DB_PATH, get_connection
from core.features.pipeline import DraftFeaturePipeline, ENRICHED_PLAYER_SPATIAL_TENSOR_DIM
from core.player_priors import PlayerPriorsTracker
from core.spatial_rnn import load_spatial_encoder_weights, spatial_encoder_id
from core.equivariant_model import build_equivariant_deepsets_model, NNRIOT_CUSTOM_OBJECTS
from core.generate_graph import LOSS_WEIGHTS, LOSS_PER_HEAD, get_adjusted_loss_weights
from core.pcgrad import create_compiled_pcgrad_step
from core.training_utils import _build_targets
from core.training_metrics import evaluate_winner
from core.calibration import fit_temperature, apply_temperature_to_probabilities
from core.selective_classification import (
    evaluate_selective_coverage,
    DEFAULT_CONVICTION_THRESHOLDS,
    format_conviction_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_spatial_deepsets")


def load_spatial_training_records(db_path: str = DB_PATH, limit: int | None = None) -> list[dict]:
    """Load training dataset records for matches that have spatial trajectory embeddings."""
    logger.info("Loading training records with spatial embeddings from %s (limit=%s)...", db_path, limit)
    with get_connection(db_path) as conn:
        query = """
            SELECT t.match_id, t.feature_json, t.labels_json, t.winner_label, m.game_creation_time
            FROM matches m
            JOIN training_dataset t ON t.match_id = m.match_id
            WHERE m.game_creation_time > 0
            ORDER BY m.game_creation_time ASC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()

    logger.info("Retrieved %d matches with spatial embeddings.", len(rows))
    records = []
    for r in rows:
        feat = _decode(r["feature_json"])
        labels = _decode(r["labels_json"])
        if not isinstance(feat, dict) or not isinstance(labels, dict):
            continue
        records.append({
            "match_id": r["match_id"],
            "game_creation_time": r["game_creation_time"],
            "feature_json": feat,
            "labels_json": labels,
            "winner_label": r["winner_label"],
        })
    logger.info("Successfully validated %d training records.", len(records))
    return records


def load_spatial_embeddings_by_puuid(db_path: str, encoder_id: str) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Map puuid -> chronological (game_creation_time, embedding) from the pinned encoder only."""
    with get_connection(db_path) as conn:
        emb_rows = conn.execute(
            """
            SELECT puuid, game_creation_time, embedding_blob
            FROM player_spatial_embeddings
            WHERE encoder_id = ?
            ORDER BY game_creation_time ASC
            """,
            (encoder_id,),
        ).fetchall()
        ignored = conn.execute(
            "SELECT COUNT(*) FROM player_spatial_embeddings WHERE encoder_id IS NOT ?", (encoder_id,)
        ).fetchone()[0]
    if not emb_rows:
        raise ValueError(
            f"{db_path} holds no spatial embeddings from encoder {encoder_id}; "
            "run scripts/compute_spatial_embeddings.py --recompute-all first"
        )
    if ignored:
        logger.warning("Ignoring %d spatial embeddings from other or unknown encoders.", ignored)

    player_embeddings_by_puuid: dict[str, list[tuple[int, np.ndarray]]] = {}
    for row in emb_rows:
        vec = np.frombuffer(row["embedding_blob"], dtype=np.float32)
        player_embeddings_by_puuid.setdefault(row["puuid"], []).append((row["game_creation_time"], vec))
    return player_embeddings_by_puuid


def prepare_spatial_dataset(
    records: list[dict],
    db_path: str = DB_PATH,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[dict, dict, dict]:
    """Extract 46D spatial tensors sequentially in chronological order to prevent data leakage."""
    logger.info("Extracting spatial tensors with chronological PlayerPriorsTracker...")

    encoder_id = spatial_encoder_id(load_spatial_encoder_weights())
    player_embeddings_by_puuid = load_spatial_embeddings_by_puuid(db_path, encoder_id)

    tracker = PlayerPriorsTracker()
    pipeline = DraftFeaturePipeline(player_tracker=tracker)

    n_total = len(records)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    n_test = n_total - n_train - n_val

    splits = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }

    tensors = {
        split: {
            "team_a": np.zeros((len(recs), 5, ENRICHED_PLAYER_SPATIAL_TENSOR_DIM), dtype=np.float32),
            "team_b": np.zeros((len(recs), 5, ENRICHED_PLAYER_SPATIAL_TENSOR_DIM), dtype=np.float32),
        }
        for split, recs in splits.items()
    }

    # Chronologically advance player history pointer in strict O(N) linear time (zero future leakage)
    puuid_ptrs: dict[str, int] = defaultdict(int)

    for split_name, recs in splits.items():
        for idx, rec in enumerate(recs):
            match_time = rec["game_creation_time"] or 0
            feature_dict = rec["feature_json"]

            for team_key in ("team_a", "team_b"):
                for p in feature_dict.get(team_key, []):
                    puuid = p.get("puuid")
                    if puuid and puuid in player_embeddings_by_puuid:
                        embs = player_embeddings_by_puuid[puuid]
                        ptr = puuid_ptrs[puuid]
                        while ptr < len(embs) and embs[ptr][0] < match_time:
                            tracker.record_spatial_embedding(puuid, embs[ptr][1])
                            ptr += 1
                        puuid_ptrs[puuid] = ptr

            # Extract 46-dimensional tensors
            ta, tb = pipeline.extract_tensors(feature_dict, include_spatial=True)
            tensors[split_name]["team_a"][idx] = ta
            tensors[split_name]["team_b"][idx] = tb

    targets = {
        split_name: _build_targets(recs)
        for split_name, recs in splits.items()
    }

    return tensors, targets, splits


def train_spatial_deepsets(
    output_dir: str = "runs/spatial_deepsets_run1",
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.0003,
    patience: int = 5,
    limit: int | None = None,
    db_path: str = DB_PATH,
    aux_weight_scale: float = 0.1,
    use_cosine_decay: bool = False,
    use_pcgrad: bool = False,
    optimizer_type: str = "adamw",
    dropout_rate: float = 0.30,
    spatial_noise_std: float = 0.05,
    label_smoothing: float = 0.05,
) -> dict:
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    records = load_spatial_training_records(db_path=db_path, limit=limit)
    if len(records) < 50:
        raise ValueError(f"Not enough records with spatial embeddings ({len(records)} found).")

    x, y, splits = prepare_spatial_dataset(records, db_path=db_path)
    y_train = y["train"]["winner"]
    train_rate = float(y_train.mean())

    n_train = len(x["train"]["team_a"])
    steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * epochs

    if use_cosine_decay:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps,
            alpha=0.01,
        )
    else:
        lr_schedule = learning_rate

    logger.info("Building Equivariant DeepSets Model with 46-dim spatial inputs...")
    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_SPATIAL_TENSOR_DIM,
        use_cross_attention=True,
        learning_rate=lr_schedule,
        use_evidential=True,
        pruned_heads=True,
        optimizer_type=optimizer_type,
        dropout_rate=dropout_rate,
        spatial_noise_std=spatial_noise_std,
        label_smoothing=label_smoothing,
    )

    loss_weights = get_adjusted_loss_weights(aux_scale=aux_weight_scale)
    loss_fns = {
        head: tf.keras.losses.get(model.loss[head]) if isinstance(model.loss, dict) and head in model.loss
        else tf.keras.losses.get(LOSS_PER_HEAD.get(head, "mse"))
        for head in LOSS_PER_HEAD
        if head in model.output_names
    }

    if use_pcgrad:
        logger.info("Using compiled PCGrad multi-task gradient deconfliction.")
        pcgrad_step = create_compiled_pcgrad_step(
            model=model,
            optimizer=model.optimizer,
            loss_fns=loss_fns,
            loss_weights=loss_weights,
            jit_compile=False,
        )

        def train_step(xa, xb, yb):
            losses = pcgrad_step({"team_a": xa, "team_b": xb}, yb)
            return tf.add_n(list(losses.values()))
    else:
        logger.info("Using standard vectorized GradientTape.")

        @tf.function
        def train_step(xa, xb, yb):
            with tf.GradientTape() as tape:
                preds = model({"team_a": xa, "team_b": xb}, training=True)
                losses = []
                for head, loss_fn in loss_fns.items():
                    if head in preds and head in yb:
                        w = loss_weights.get(head, 1.0)
                        losses.append(w * tf.reduce_mean(loss_fn(yb[head], preds[head])))
                batch_loss = tf.add_n(losses)
            grads = tape.gradient(batch_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return batch_loss

    logger.info("Starting training on %d samples (%d val, %d test)...", n_train, len(x["val"]["team_a"]), len(x["test"]["team_a"]))

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience = 5
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = np.random.permutation(n_train)
        xa_shuff = x["train"]["team_a"][perm]
        xb_shuff = x["train"]["team_b"][perm]
        y_shuff = {k: tf.constant(v[perm]) for k, v in y["train"].items() if k in model.output_names}

        epoch_loss = 0.0
        seen = 0

        for start in range(0, n_train, batch_size):
            xa_batch = tf.constant(xa_shuff[start:start + batch_size])
            xb_batch = tf.constant(xb_shuff[start:start + batch_size])
            yb_batch = {k: y_shuff[k][start:start + batch_size] for k in y_shuff}

            batch_loss = train_step(xa_batch, xb_batch, yb_batch)

            epoch_loss += float(batch_loss.numpy()) * len(xa_batch)
            seen += len(xa_batch)

        train_loss = epoch_loss / max(1, seen)
        epoch_time = time.time() - t0

        # Validation pass
        val_preds = model(x["val"], training=False)
        val_losses = []
        for head, loss_fn in loss_fns.items():
            if head in val_preds and head in y["val"]:
                w = loss_weights.get(head, 1.0)
                term_loss = tf.reduce_mean(loss_fn(y["val"][head], val_preds[head]))
                val_losses.append(w * term_loss)
        val_loss = float(tf.add_n(val_losses).numpy()) if val_losses else 0.0

        val_winner_probs = val_preds["winner"].numpy()[:, 1]
        val_winner_labels = np.argmax(y["val"]["winner"], axis=1)
        val_acc = float(np.mean((val_winner_probs >= 0.5) == val_winner_labels))

        history.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 5),
            "val_loss": round(float(val_loss), 5),
            "val_acc": round(float(val_acc), 5),
            "epoch_time_s": round(float(epoch_time), 2),
        })

        bias_info = ""
        try:
            side_bias_layer = model.get_layer("winner_side_bias")
            if side_bias_layer is not None:
                b_val = float(side_bias_layer.bias.numpy()[0])
                implied_blue = 1.0 / (1.0 + np.exp(2.0 * b_val))
                bias_info = f", blue_bias={b_val:+.4f} (implied_blue_wr={implied_blue*100:.1f}%)"
        except Exception:
            pass

        logger.info(
            "Epoch %02d/%02d (%.1fs) - Loss: %.4f - Val Loss: %.4f - Val Win Acc: %.4f%s",
            epoch, epochs, epoch_time, train_loss, val_loss, val_acc, bias_info,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = model.get_weights()
            logger.info("  * New best validation loss (%.4f) -> saved checkpoint weights", val_loss)
        elif patience is not None and (epoch - best_epoch) >= patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d exceeded).", epoch, patience)
            break

    if best_weights:
        model.set_weights(best_weights)

    # Save model checkpoint
    model_path = out_path / "model.keras"
    model.save(model_path)
    logger.info("Saved trained spatial model checkpoint to %s", model_path)

    # Calibration & Test Evaluation
    val_raw_probs = model(x["val"], training=False)["winner"].numpy()[:, 1]
    test_raw_probs = model(x["test"], training=False)["winner"].numpy()[:, 1]
    val_winner_labels = np.argmax(y["val"]["winner"], axis=1)
    test_winner_labels = np.argmax(y["test"]["winner"], axis=1)

    cal_artifact = fit_temperature(val_raw_probs, val_winner_labels)
    temperature = float(cal_artifact["temperature"])
    test_cal_probs = apply_temperature_to_probabilities(test_raw_probs, temperature)

    test_eval_raw = evaluate_winner(test_winner_labels, test_raw_probs, train_rate, diagnostics=True)
    test_eval_cal = evaluate_winner(test_winner_labels, test_cal_probs, train_rate, diagnostics=True)

    stratification = {
        format_conviction_key(tau): evaluate_selective_coverage(test_winner_labels, test_cal_probs, tau)
        for tau in DEFAULT_CONVICTION_THRESHOLDS
    }

    metrics_report = {
        "model_type": "equivariant_deepsets_spatial_dual_gru",
        "player_feature_dim": ENRICHED_PLAYER_SPATIAL_TENSOR_DIM,
        "n_samples": len(records),
        "temperature": round(float(temperature), 4),
        "history": history,
        "test_eval_raw": test_eval_raw,
        "test_eval_cal": test_eval_cal,
        "selective_stratification": stratification,
    }

    metrics_file = out_path / "model.metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    logger.info("Saved metrics report to %s", metrics_file)
    model_metrics = test_eval_cal.get("model", test_eval_cal)
    acc = model_metrics.get("accuracy", 0.0)
    brier = model_metrics.get("brier", 0.0)
    logger.info("Final Test Accuracy (Calibrated): %.4f (Brier: %.4f)", acc, brier)
    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Equivariant DeepSets on spatial features.")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to SQLite database")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate (default: 0.0003)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples")
    parser.add_argument("--output-dir", type=str, default="runs/spatial_deepsets_run1", help="Output directory")
    parser.add_argument("--use-cosine-decay", action="store_true", help="Enable CosineDecay learning rate schedule")
    parser.add_argument("--aux-weight-scale", type=float, default=0.1, help="Scale factor for auxiliary regression head loss weights (default: 0.1)")
    parser.add_argument("--use-pcgrad", action="store_true", help="Enable PCGrad multi-task gradient deconfliction")
    args = parser.parse_args()

    train_spatial_deepsets(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        limit=args.limit,
        db_path=args.db,
        aux_weight_scale=args.aux_weight_scale,
        use_cosine_decay=args.use_cosine_decay,
        use_pcgrad=args.use_pcgrad,
    )
