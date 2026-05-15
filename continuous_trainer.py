#!/usr/bin/env python3
"""
continuous_trainer.py - Automated training loop for NNriot
Runs every 10 minutes to process new matches and tag them as trained.
"""

import os

# Silence TensorFlow oneDNN performance warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import random
import tensorflow as tf
import numpy as np
import time
import logging
import database
import json_utils
from generate_graph import build_keras_model

# Configuration
VECTOR_DIM = 100000
TRAINING_INTERVAL_SECONDS = 10 * 60  # 10 minutes
BATCH_SIZE = 500
EPOCHS_PER_BATCH = 2  # Reduced to avoid overfitting

# Use script-relative paths so the trainer works regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model.keras")

# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def sparse_microbatch_generator(x_sparse, y_dense, batch_size=32):
    """Yields small dense micro-batches from a large sparse matrix to save memory."""
    n_samples = x_sparse.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        x_batch = x_sparse[batch_indices].toarray()
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
                self.model = build_keras_model(input_dim=VECTOR_DIM)
        else:
            self.model = build_keras_model(input_dim=VECTOR_DIM)
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

        logger.info(
            "Found %d new matches. Adding %d historical matches for replay (Total batch: %d).",
            len(new_records),
            len(old_records),
            len(records),
        )

        random.shuffle(records)

        # Prepare inputs natively via feature hasher which uses iterators & small memory overhead
        json_features = [r["feature_json"] for r in records]
        x_data_sparse = json_utils.json_to_vector(json_features, dim=VECTOR_DIM)

        # Prepare labels (0 -> [1, 0], 1 -> [0, 1])
        y_data = np.zeros((len(records), 2), dtype=np.float32)
        for i, r in enumerate(records):
            if r["winner_label"] == 0:
                y_data[i] = [1.0, 0.0]
            else:
                y_data[i] = [0.0, 1.0]

        logger.info(
            "Starting %d epochs with micro-batch optimization...", EPOCHS_PER_BATCH
        )
        start_time = time.time()

        # Micro-batch training loop
        final_loss = 0.0
        for epoch in range(EPOCHS_PER_BATCH):
            batch_losses = []
            for x_batch, y_batch in sparse_microbatch_generator(
                x_data_sparse, y_data, batch_size=64
            ):
                loss, acc = self.model.train_on_batch(x_batch, y_batch)
                batch_losses.append(loss)
            final_loss = np.mean(batch_losses)
            logger.info(
                "Epoch %d/%d completed. Loss: %.6f",
                epoch + 1,
                EPOCHS_PER_BATCH,
                final_loss,
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

    def predict(self, x_dense: np.ndarray) -> np.ndarray:
        """
        Run inference and return softmax probabilities.

        Parameters
        ----------
        x_dense:
            Dense NumPy array of shape (n_samples, VECTOR_DIM).

        Returns
        -------
        np.ndarray of shape (n_samples, 2) — [p_team_a_win, p_team_b_win]
        """
        return self.model.predict(x_dense, verbose=0)


def main():
    database.migrate_db()

    trainer = ContinuousTrainer()

    print("\n" + "=" * 40)
    print("🚀 NNriot Batch Trainer Started")
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
            logger.error("Error in training batch: %s", e)

        # Sleep before next cycle
        logger.info(
            "Sleeping for %d minutes before next check...",
            TRAINING_INTERVAL_SECONDS // 60,
        )
        time.sleep(TRAINING_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
