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
from feature_labels import LABEL_KEYS, ALL_LABEL_KEYS, VECTOR_DIM
from generate_graph import build_multi_output_model

# Configuration
TRAINING_INTERVAL_SECONDS = 10 * 60  # 10 minutes

# Normalization stats for regression heads, derived empirically from the
# production training set (47k Match-V5 records). Values are (mean, std).
# After normalization, each regression head sees ~unit-variance targets,
# letting LOSS_WEIGHTS for regressions be 1.0 alongside categorical heads.
REGRESSION_STATS: dict[str, tuple[float, float]] = {
    "total_kills":   (30.0, 12.0),
    "team_a_kills":  (15.0, 7.0),
    "team_b_kills":  (15.0, 7.0),
    "kill_handicap": (0.0, 12.0),
    "total_barons":  (1.0, 1.0),
    "total_dragons": (2.5, 1.5),
    "total_towers":  (10.0, 5.0),
}
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
    """Build a dict of {head_name: ndarray} ready for model.train_on_batch.

    Assumes all records have complete labels_json (filtered by caller).
    The ``.get(name, 0)`` defensive accesses below remain to protect against
    future drift, but the caller (``run_training_step``) is responsible for
    filtering incomplete records before invoking this function — otherwise
    the produced targets would not align with the input matrix built from
    the same record list.
    """
    n = len(records)
    # Pre-allocate arrays per head type
    targets: dict[str, np.ndarray] = {}

    # 2-class softmax heads: shape (n, 2), one-hot
    for name in ("winner", "team_b_kill_lead"):
        arr = np.zeros((n, 2), dtype=np.float32)
        for i, r in enumerate(records):
            val = int(r["labels_json"].get(name, 0))
            arr[i, val] = 1.0
        targets[name] = arr

    # 3-class softmax heads: shape (n, 3), one-hot
    # NOTE: timeline kill-threshold heads (first_to_N_kills) default to class 2
    # ("neither team reached the threshold") when the key is missing from a
    # record's labels_json. This makes timeline labels OPTIONAL — existing rows
    # without timeline ingestion still train these heads (toward "neither").
    for name in ("first_blood", "first_baron", "first_inhibitor", "first_tower"):
        arr = np.zeros((n, 3), dtype=np.float32)
        for i, r in enumerate(records):
            val = int(r["labels_json"].get(name, 0))
            arr[i, val] = 1.0
        targets[name] = arr
    for name in ("first_to_5_kills", "first_to_10_kills",
                 "first_to_15_kills", "first_to_20_kills"):
        arr = np.zeros((n, 3), dtype=np.float32)
        for i, r in enumerate(records):
            val = int(r["labels_json"].get(name, 2))  # default: "neither" class
            arr[i, val] = 1.0
        targets[name] = arr

    # Binary sigmoid heads: shape (n, 1), 0.0 or 1.0
    for name in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon"):
        arr = np.zeros((n, 1), dtype=np.float32)
        for i, r in enumerate(records):
            arr[i, 0] = float(r["labels_json"].get(name, 0))
        targets[name] = arr

    # Regression heads: shape (n, 1), normalized via (raw - mean) / std
    # so each head sees ~unit-variance targets (REGRESSION_STATS).
    for name in ("kill_handicap", "total_kills", "team_a_kills", "team_b_kills",
                 "total_barons", "total_dragons", "total_towers"):
        arr = np.zeros((n, 1), dtype=np.float32)
        mean, std = REGRESSION_STATS[name]
        for i, r in enumerate(records):
            raw = float(r["labels_json"].get(name, 0))
            arr[i, 0] = (raw - mean) / std
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
        """Load model_v2.keras if compatible with current architecture, else rebuild.

        ``tf.keras.models.load_model`` happily loads a legacy checkpoint with a
        different input dimension or head set — the mismatch only surfaces later
        at predict time as a confusing shape error. We validate the candidate
        eagerly and rebuild from scratch on any mismatch.
        """
        if os.path.exists(CHECKPOINT_PATH):
            try:
                candidate = tf.keras.models.load_model(CHECKPOINT_PATH)
                self._validate_checkpoint(candidate)
                self.model = candidate
                logger.info("Model restored from %s.", CHECKPOINT_PATH)
                return
            except Exception as e:
                logger.warning(
                    "Checkpoint at %s incompatible (%s). Rebuilding from scratch.",
                    CHECKPOINT_PATH, e,
                )
        self.model = build_multi_output_model(input_dim=VECTOR_DIM)
        logger.info("Model initialized from scratch.")

    def _validate_checkpoint(self, model):
        """Raise if the candidate model's architecture doesn't match the expected one.

        Checks two things:
        - Input dimension equals :data:`VECTOR_DIM` (currently 20000).
        - Output head names exactly match :data:`feature_labels.ALL_LABEL_KEYS`
          (the 22 multi-output heads). A stale 18-head checkpoint or legacy
          single-output model is rejected.
        """
        from feature_labels import ALL_LABEL_KEYS

        actual_input_dim = model.input_shape[-1]
        if actual_input_dim != VECTOR_DIM:
            raise ValueError(
                f"Checkpoint input dim {actual_input_dim} != expected {VECTOR_DIM}"
            )

        actual_heads = set(model.output_names)
        expected_heads = set(ALL_LABEL_KEYS)
        if actual_heads != expected_heads:
            missing = expected_heads - actual_heads
            extra = actual_heads - expected_heads
            raise ValueError(
                f"Checkpoint heads mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

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
        json_features = [r["feature_json"] for r in records]
        x_data_sparse = json_utils.json_to_vector(json_features, dim=VECTOR_DIM)

        # Build dict of {head_name: ndarray} targets for all 22 heads
        # (18 LABEL_KEYS + 4 TIMELINE_LABEL_KEYS; the latter default to class 2
        # when the row's labels_json is missing them).
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
