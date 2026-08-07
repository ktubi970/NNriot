#!/usr/bin/env python3
"""
sanity_logreg.py - Localize the cause of poor model performance.

Trains a plain logistic regression on the exact same feature vectors the NN
uses (json_utils.json_to_vector, dim=VECTOR_DIM) to predict `winner`. Uses
a clean train/test split so the test number is a real held-out estimate.

If logreg >> 0.57, the NN training pipeline is the problem.
If logreg ~= 0.57, the features are the problem.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import time

import numpy as np

import database
import json_utils
from feature_labels import VECTOR_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sanity_logreg")

N_SAMPLES = 5000
TEST_FRACTION = 0.2
SEED = 42


def _fetch_records(n: int) -> list[dict]:
    """Mix untrained holdout + trained replay so we have enough rows."""
    rows = database.get_random_trained_records(limit=n)
    return [r for r in rows if r.get("feature_json") and r.get("winner_label") is not None]


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    logger.info("Fetching up to %d records ...", N_SAMPLES)
    records = _fetch_records(N_SAMPLES)
    logger.info("Got %d usable records.", len(records))
    if len(records) < 200:
        raise SystemExit("Not enough records for a meaningful split.")

    json_features = [r["feature_json"] for r in records]
    y = np.asarray([int(r["winner_label"]) for r in records])

    t0 = time.time()
    x_sparse = json_utils.json_to_vector(json_features, dim=VECTOR_DIM)
    logger.info(
        "Vectorized: shape=%s nnz=%d (%.2fs)",
        x_sparse.shape, x_sparse.nnz, time.time() - t0,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_sparse, y, test_size=TEST_FRACTION, random_state=SEED, stratify=y,
    )
    logger.info("Train: %d  Test: %d", x_train.shape[0], x_test.shape[0])
    logger.info("Train winner rate: %.3f  Test winner rate: %.3f", y_train.mean(), y_test.mean())

    t0 = time.time()
    clf = LogisticRegression(
        solver="liblinear",  # handles sparse, L2 by default
        max_iter=200,
        C=1.0,
        random_state=SEED,
    )
    clf.fit(x_train, y_train)
    logger.info("Fit done in %.2fs.", time.time() - t0)

    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    print()
    print("=== Sanity check: logistic regression on `winner` ===")
    print(f"Features: hashed JSON, dim={VECTOR_DIM}")
    print(f"Train n={x_train.shape[0]}   Test n={x_test.shape[0]}")
    print(f"Train accuracy:    {train_acc:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")
    print(f"Majority baseline: {baseline:.4f}")
    print(f"Lift on test:      {test_acc - baseline:+.4f}")
    print()
    if test_acc - baseline > 0.05:
        print("-> Features carry signal. The NN training pipeline is the problem.")
    elif test_acc - baseline > 0.01:
        print("-> Features carry weak signal. NN may be wasting it, but features need work too.")
    else:
        print("-> Features carry no signal. The hashing+JSON pipeline is destroying information.")


if __name__ == "__main__":
    main()
