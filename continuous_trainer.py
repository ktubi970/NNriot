#!/usr/bin/env python3
"""
continuous_trainer.py - Automated training loop for NNriot
Runs every 10 minutes to process new matches and tag them as trained.
"""

import os

# Silence TensorFlow oneDNN performance warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import tensorflow as tf
import numpy as np
import time
import logging
import database
import json_utils
from generate_graph import build_multi_output_model

# Configuration
VECTOR_DIM = 100000
TRAINING_INTERVAL_SECONDS = 10 * 60  # 10 minutes
# BATCH_SIZE / EPOCHS_PER_BATCH can be overridden via env vars so users can
# do a small smoke run before kicking off a full training cycle:
#   NNRIOT_BATCH_SIZE=100 NNRIOT_EPOCHS=1 python continuous_trainer.py
BATCH_SIZE = int(os.environ.get("NNRIOT_BATCH_SIZE", "500"))
EPOCHS_PER_BATCH = int(os.environ.get("NNRIOT_EPOCHS", "2"))  # Reduced to avoid overfitting

# Use script-relative paths so the trainer works regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_v2.keras")

# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _build_targets(records: list[dict]) -> dict[str, np.ndarray]:
    """
    Given training records (each with a parsed `labels_json` dict),
    build a dict of {head_name: ndarray} ready for model.train_on_batch.
    """
    n = len(records)
    # Pre-allocate arrays per head type
    targets: dict[str, np.ndarray] = {}

    # 2-class softmax heads: shape (n, 2), one-hot
    for name in ("winner", "winner_kills"):
        arr = np.zeros((n, 2), dtype=np.float32)
        for i, r in enumerate(records):
            val = int(r["labels_json"][name])
            arr[i, val] = 1.0
        targets[name] = arr

    # 3-class softmax heads: shape (n, 3), one-hot
    for name in ("first_blood", "first_baron", "first_inhibitor", "first_tower"):
        arr = np.zeros((n, 3), dtype=np.float32)
        for i, r in enumerate(records):
            val = int(r["labels_json"][name])
            arr[i, val] = 1.0
        targets[name] = arr

    # Binary sigmoid heads: shape (n, 1), 0.0 or 1.0
    for name in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon"):
        arr = np.zeros((n, 1), dtype=np.float32)
        for i, r in enumerate(records):
            arr[i, 0] = float(r["labels_json"][name])
        targets[name] = arr

    # Regression heads: shape (n, 1), raw float
    for name in ("kill_handicap", "total_kills", "team_a_kills", "team_b_kills",
                 "total_barons", "total_dragons", "total_towers"):
        arr = np.zeros((n, 1), dtype=np.float32)
        for i, r in enumerate(records):
            arr[i, 0] = float(r["labels_json"][name])
        targets[name] = arr

    return targets


def sparse_microbatch_generator(x_sparse, y_dense, batch_size=32):
    """Yields small dense micro-batches from a large sparse matrix to save memory.

    ``y_dense`` may be either a plain ndarray (legacy) or a dict mapping head
    names to ndarrays (multi-output). The generator preserves whichever shape
    was passed in.
    """
    n_samples = x_sparse.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        x_batch = x_sparse[batch_indices].toarray()
        if isinstance(y_dense, dict):
            y_batch = {k: v[batch_indices] for k, v in y_dense.items()}
        else:
            y_batch = y_dense[batch_indices]
        yield x_batch, y_batch


class ContinuousTrainer:
    def __init__(self):
        self._initialize_model()

    def _initialize_model(self):
        """Initialize or restore the model."""
        if os.path.exists(CHECKPOINT_PATH):
            try:
                self.model = tf.keras.models.load_model(CHECKPOINT_PATH)
                logger.info("Model restored from %s.", CHECKPOINT_PATH)
            except Exception as e:
                logger.warning("Failed to restore checkpoint: %s. Re-initializing.", e)
                self.model = build_multi_output_model(input_dim=VECTOR_DIM)
        else:
            self.model = build_multi_output_model(input_dim=VECTOR_DIM)
            logger.info("No checkpoint found. Model initialized from scratch.")

    def close(self):
        """No explicit session close needed in TF2."""
        logger.info("Trainer safely closed.")

    def run_training_step(self):
        """Fetch new data, train, and tag. Returns number of records processed."""
        logger.info("Checking for new training data...")
        new_records = database.get_untrained_records(limit=BATCH_SIZE)

        if not new_records:
            logger.info("No new data to train on.")
            return 0

        # Experience Replay: Fetch random historical records to mix with new ones
        # Increased ratio to 2x historical data relative to new_records
        old_records = database.get_random_trained_records(limit=len(new_records) * 2)
        records = new_records + old_records
        records = [r for r in records if r["feature_json"] is not None]

        logger.info(
            "Found %d new matches. Adding %d historical matches for replay (Total batch: %d).",
            len(new_records),
            len(old_records),
            len(records),
        )

        random.shuffle(records)

        # P3: multi-output training requires labels_json. Skip records that
        # haven't been backfilled (legacy rows from before the schema bump).
        records = [r for r in records if r["labels_json"] is not None]
        if not records:
            logger.info("All available records lack labels_json; skipping.")
            return 0

        # Prepare inputs natively via feature hasher which uses iterators & small memory overhead
        json_features = [r["feature_json"] for r in records]
        x_data_sparse = json_utils.json_to_vector(json_features, dim=VECTOR_DIM)

        # Build dict of {head_name: ndarray} targets for all 18 heads
        y_data = _build_targets(records)

        logger.info(
            "Starting %d epochs with micro-batch optimization...", EPOCHS_PER_BATCH
        )
        start_time = time.time()

        # Micro-batch training loop
        final_loss = 0.0
        losses: dict = {}
        for epoch in range(EPOCHS_PER_BATCH):
            batch_losses = []
            for x_batch, y_batch in sparse_microbatch_generator(
                x_data_sparse, y_data, batch_size=64
            ):
                losses = self.model.train_on_batch(
                    x_batch, y_batch, return_dict=True
                )
                batch_losses.append(losses["loss"])  # total loss
            final_loss = float(np.mean(batch_losses))
            if losses:
                logger.info(
                    "Epoch %d/%d done. total_loss=%.4f winner_acc=%.3f total_kills_mae=%.2f elder_dragon_acc=%.3f",
                    epoch + 1,
                    EPOCHS_PER_BATCH,
                    final_loss,
                    losses.get("winner_accuracy", 0),
                    losses.get("total_kills_mae", 0),
                    losses.get("elder_dragon_accuracy", 0),
                )

        duration = time.time() - start_time
        logger.info(
            "Batch complete. Final Loss: %.6f. Time: %.2fs", final_loss, duration
        )

        # Tag ONLY the new records as trained
        new_record_ids = [r["id"] for r in new_records]
        database.mark_as_trained(new_record_ids)
        logger.info("Tagged %d new records as trained in DB.", len(new_record_ids))

        # Save model natively
        self.model.save(CHECKPOINT_PATH)
        logger.info("Model checkpoint saved to %s.", CHECKPOINT_PATH)

        return len(new_records)

    def predict(self, x_dense: np.ndarray):
        """
        Run inference and return per-head outputs.

        Parameters
        ----------
        x_dense:
            Dense NumPy array of shape (n_samples, VECTOR_DIM).

        Returns
        -------
        dict[str, np.ndarray]
            One entry per output head; e.g. ``out["winner"]`` has shape
            (n_samples, 2) and contains softmax probabilities. Regression
            heads return shape (n_samples, 1).

        Notes
        -----
        Return type changed from ndarray to dict in P3 (multi-output model).
        Callers that previously did ``preds[0]`` for the single softmax row
        must be updated to use ``preds["winner"][0]``. This is a known
        incompatibility for ``final_web_app.py`` to be fixed in P4.
        """
        return self.model.predict(x_dense, verbose=0)


def main():
    database.migrate_db()

    trainer = ContinuousTrainer()

    print("\n" + "=" * 40)
    print("NNriot Batch Trainer Started")
    print("Processing all untrained data in database...")
    print("=" * 40 + "\n")

    total_processed = 0
    batch_count = 0

    # Loop indefinitely, checking for new data every TRAINING_INTERVAL_SECONDS
    while True:
        try:
            records_processed = trainer.run_training_step()
            if records_processed > 0:
                batch_count += 1
                total_processed += records_processed
                logger.info(
                    "Batch %d completed. Records processed: %d. Total so far: %d",
                    batch_count,
                    records_processed,
                    total_processed,
                )
                # Immediately try again to drain any remaining data
                continue
            else:
                logger.info(
                    "No new data available. Will recheck in %d minutes.",
                    TRAINING_INTERVAL_SECONDS // 60,
                )
        except Exception as e:
            logger.error("Error in training batch: %s", e, exc_info=True)

        # Sleep before next cycle
        logger.info(
            "Sleeping for %d minutes before next check...",
            TRAINING_INTERVAL_SECONDS // 60,
        )
        time.sleep(TRAINING_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
