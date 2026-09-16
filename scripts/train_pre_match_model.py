#!/usr/bin/env python3
"""
train_pre_match_model.py - Calibrate and train pre-match baseline model.

Fetches matches from training_data.db, sanitizes features to strictly pre-match
signals (champions, roles, game_mode, game_version, season_id, region, career stats)
with zero post-game metric leakage, vectorizes them into VECTOR_DIM (20,000),
builds a multi-output model, and saves the calibrated model to scripts/model_v2.keras.
"""

import os
import sys

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import logging
import random
import time
import numpy as np
import tensorflow as tf

from core import database
from core import json_utils
from core.feature_labels import LABEL_KEYS, VECTOR_DIM
from core.generate_graph import build_multi_output_model
from scripts.continuous_trainer import _build_targets, sparse_microbatch_generator

DEFAULT_CHECKPOINT = os.path.join(BASE_DIR, "model_v2.keras")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_pre_match_model")


def to_pre_match_features(feat: dict) -> dict:
    """
    Extract strictly pre-match features, stripping all post-match leakage
    (kills, deaths, assists, gold, kda, etc.).
    """
    if not isinstance(feat, dict):
        return {}

    # If raw match format with participants
    if "participants" in feat or (
        "info" in feat
        and isinstance(feat.get("info"), dict)
        and "participants" in feat["info"]
    ):
        return json_utils.extract_pre_match_features(feat)

    # Standard matchup format with team_a and team_b
    clean = {
        "game_mode": feat.get("game_mode", "CLASSIC"),
        "game_version": feat.get("game_version", "UNKNOWN"),
        "season_id": feat.get("season_id"),
    }
    if "region" in feat and feat["region"]:
        clean["region"] = feat["region"]

    for team_key in ("team_a", "team_b"):
        clean_team = []
        for p in feat.get(team_key, []):
            p_clean = {
                "champion": p.get("champion")
                or p.get("championName")
                or p.get("champion_name", "UNKNOWN"),
                "role": p.get("role") or p.get("teamPosition") or "UNKNOWN",
            }
            if "career_win_rate" in p:
                p_clean["career_win_rate"] = float(p["career_win_rate"])
            if "career_matches" in p:
                p_clean["career_matches"] = int(p["career_matches"])
            clean_team.append(p_clean)
        clean[team_key] = clean_team
    return clean


def fetch_training_records(limit: int = 5000) -> list[dict]:
    """Fetch training records from the database that have valid labels."""
    logger.info("Querying training records (limit=%d)...", limit)
    with database.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, match_id, feature_json, winner_label, labels_json
            FROM training_dataset
            WHERE labels_json IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    required = set(LABEL_KEYS)
    valid = []
    for r in rows:
        d = dict(r)
        d["feature_json"] = database._decode_compressed_json(d["feature_json"])
        d["labels_json"] = database._decode_compressed_json(d["labels_json"])
        if not d["feature_json"] or not d.get("labels_json"):
            continue
        if not required.issubset(d["labels_json"].keys()):
            continue
        valid.append(d)

    logger.info("Found %d valid records with complete labels.", len(valid))
    return valid


def train_model(
    records: list[dict],
    epochs: int = 5,
    batch_size: int = 64,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    dim: int = VECTOR_DIM,
) -> tf.keras.Model:
    """Train multi-output model on pre-match features and save checkpoint."""
    # 1. Clean features
    logger.info("Sanitizing %d records to strictly pre-match features...", len(records))
    pre_match_features = [to_pre_match_features(r["feature_json"]) for r in records]

    # 2. Vectorize features
    logger.info("Vectorizing features to dim=%d...", dim)
    t0 = time.time()
    x_sparse = json_utils.json_to_vector(pre_match_features, dim=dim)
    y_targets = _build_targets(records)
    logger.info(
        "Vectorization complete in %.2fs. Shape: %s, nnz: %d",
        time.time() - t0,
        x_sparse.shape,
        x_sparse.nnz,
    )

    # 3. Build model
    logger.info("Instantiating build_multi_output_model(input_dim=%d)...", dim)
    model = build_multi_output_model(input_dim=dim)
    logger.info(
        "Model built. Input shape: %s, output heads: %d",
        model.input_shape,
        len(model.output_names),
    )

    # 4. Train
    logger.info("Starting training for %d epochs (batch_size=%d)...", epochs, batch_size)
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        batch_losses = []
        last_metrics = {}
        for x_batch, y_batch in sparse_microbatch_generator(
            x_sparse, y_targets, batch_size=batch_size
        ):
            metrics = model.train_on_batch(x_batch, y_batch, return_dict=True)
            batch_losses.append(metrics["loss"])
            last_metrics = metrics

        avg_loss = float(np.mean(batch_losses))
        winner_acc = last_metrics.get("winner_accuracy", 0.0)
        total_kills_mae = last_metrics.get("total_kills_mae", 0.0)
        logger.info(
            "Epoch %d/%d - Loss: %.4f - Winner Acc: %.3f - Total Kills MAE: %.2f (%.2fs)",
            epoch,
            epochs,
            avg_loss,
            winner_acc,
            total_kills_mae,
            time.time() - t_epoch,
        )

    # 5. Save model
    logger.info("Saving calibrated model to %s...", checkpoint_path)
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    model.save(checkpoint_path)
    logger.info(
        "Model successfully saved to %s (size: %.1f MB)",
        checkpoint_path,
        os.path.getsize(checkpoint_path) / (1024 * 1024),
    )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train pre-match baseline model calibrated to 20,000 dimensions."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of records to fetch for training (default: 5000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Micro-batch size (default: 64)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=VECTOR_DIM,
        help="Input vector dimension (default: 20000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Output checkpoint path (default: scripts/model_v2.keras)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    records = fetch_training_records(limit=args.samples)
    if not records:
        logger.error("No valid training records found in database.")
        sys.exit(1)

    random.shuffle(records)
    train_model(
        records=records,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=args.output,
        dim=args.dim,
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
