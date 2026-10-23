#!/usr/bin/env python3
"""
continuous_trainer.py - Automated training loop for NNriot
Runs every 10 minutes to process new matches and tag them as trained.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

# Silence TensorFlow oneDNN performance warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import threading
import tensorflow as tf
import numpy as np
import time
import logging
from core import config, database
from core.repository import TrainingRepository
from core import json_utils
from core.feature_labels import LABEL_KEYS, ALL_LABEL_KEYS, VECTOR_DIM
from core.pre_match_dataset import to_pre_match_features
from core.training_utils import REGRESSION_STATS, _build_targets, sparse_microbatch_generator
from core.generate_graph import build_multi_output_model
from core.calibration import (
    UncalibratedModelError,
    apply_temperature_to_probabilities,
    invalidate_calibration,
)

# Configuration
TRAINING_INTERVAL_SECONDS = 10 * 60  # 10 minutes

# Continuous updates remain available through cli.py train-continuous.
# Candidate evaluation uses the chronological pre-match training command.
BATCH_SIZE = int(os.environ.get("NNRIOT_BATCH_SIZE", "3000"))
EPOCHS_PER_BATCH = int(os.environ.get("NNRIOT_EPOCHS", "15"))

# Use script-relative paths so the trainer works regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.environ.get("NNRIOT_CHECKPOINT_PATH") or os.path.join(BASE_DIR, "model_v2.keras")

# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class ContinuousTrainer:
    def __init__(self, checkpoint_path: str = None, db_path: str = None):
        self._model_lock = threading.RLock()
        from core.model_factory import ModelFactory
        target_checkpoint = checkpoint_path or CHECKPOINT_PATH
        self.model, self.checkpoint_path = ModelFactory.get_model(
            version="v2", 
            input_dim=VECTOR_DIM, 
            force_rebuild=False, 
            checkpoint_path=target_checkpoint
        )
        self.repo = TrainingRepository(db_path=db_path)

    def close(self):
        """No explicit session close needed in TF2."""
        logger.info("Trainer safely closed.")

    def run_training_step(self):
        """Fetch new data, train, and tag. Returns number of records processed."""
        with self._model_lock:
            return self._run_training_step_locked()

    def _run_training_step_locked(self):
        logger.info("Checking for new training data...")
        new_records = self.repo.get_untrained_records(limit=BATCH_SIZE)

        if not new_records:
            logger.info("No new data to train on.")
            return 0

        # Experience Replay: Fetch random historical records to mix with new ones
        # Increased ratio to 2x historical data relative to new_records
        old_records = self.repo.get_random_trained_records(limit=len(new_records) * 2)
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

        # Filter records with incomplete labels_json before building inputs/targets.
        # Must happen BEFORE x_data_sparse is built so that x.shape[0] matches
        # the length of each per-head target array produced by _build_targets.
        # Filter uses LABEL_KEYS (18) — timeline keys (TIMELINE_LABEL_KEYS) are
        # OPTIONAL and default to class 2 in _build_targets when absent.
        required_keys = set(LABEL_KEYS)
        incomplete_count = sum(
            1 for r in records if not required_keys.issubset(r["labels_json"].keys())
        )
        if incomplete_count:
            logger.warning(
                "Skipping %d records with incomplete labels_json (missing keys)",
                incomplete_count,
            )
            records = [
                r for r in records if required_keys.issubset(r["labels_json"].keys())
            ]
            if not records:
                logger.info("All records have incomplete labels; skipping training step.")
                return 0

        # Prepare inputs natively via feature hasher which uses iterators & small memory overhead
        # Sanitize to pre-match features to prevent any post-match leakage (kda, gold)
        json_features = [to_pre_match_features(r["feature_json"]) for r in records]
        x_data_sparse = json_utils.json_to_vector(json_features, dim=VECTOR_DIM)

        # Build dict of {head_name: ndarray} targets for all 22 heads
        # (18 LABEL_KEYS + 4 TIMELINE_LABEL_KEYS; the latter default to class 2
        # when the row's labels_json is missing them).
        y_data = _build_targets(records)

        invalidate_calibration(self.model, self.checkpoint_path)

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
        self.repo.mark_as_trained(new_record_ids)
        logger.info("Tagged %d new records as trained in DB.", len(new_record_ids))

        # Save model natively
        self.model.save(self.checkpoint_path)
        logger.info("Model checkpoint saved to %s.", self.checkpoint_path)

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
            (n_samples, 2) and contains calibrated softmax probabilities. Regression
            heads return shape (n_samples, 1).

        Notes
        -----
        Return type changed from ndarray to dict in P3 (multi-output model).
        Callers that previously did ``preds[0]`` for the single softmax row
        must be updated to use ``preds["winner"][0]``. The web application formats these named outputs for its prediction API.
        """
        with self._model_lock:
            if getattr(self.model, 'is_calibrated', False) is not True:
                raise UncalibratedModelError('No verified winner calibration is available')
            temperature = self.model.calibration_temperature
            raw = self.model.predict(x_dense, verbose=0)
            winner = np.asarray(raw['winner'], dtype=float)
            if (winner.ndim != 2 or winner.shape[1] != 2 or
                    not np.isfinite(winner).all() or (winner < 0).any() or
                    (winner > 1).any() or not np.allclose(winner.sum(axis=1), 1.0, atol=1e-6)):
                raise ValueError('winner output must contain normalized [A, B] probabilities')
            p_b = apply_temperature_to_probabilities(winner[:, 1], temperature)
            output = dict(raw)
            output['winner'] = np.column_stack((1.0 - p_b, p_b))
            return output


def main(db_path=None):
    db_path = config.get_db_path(db_path, require_exists=True)
    database.migrate_db(db_path)

    trainer = ContinuousTrainer(db_path=db_path)

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
