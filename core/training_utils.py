"""Shared NumPy training preparation, independent of TensorFlow and storage."""

import numpy as np

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
