import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from generate_graph import (
    build_multi_output_model,
    LOSS_PER_HEAD,
    LOSS_WEIGHTS,
    METRICS_PER_HEAD,
)
from feature_labels import LABEL_KEYS


def test_constants_cover_label_keys():
    """LOSS_PER_HEAD / LOSS_WEIGHTS / METRICS_PER_HEAD must cover exactly LABEL_KEYS."""
    assert set(LOSS_PER_HEAD) == set(LABEL_KEYS)
    assert set(LOSS_WEIGHTS) == set(LABEL_KEYS)
    assert set(METRICS_PER_HEAD) == set(LABEL_KEYS)


def test_model_builds_with_small_input_dim():
    """Build the model with a small input dim to make the test fast and assert structure."""
    model = build_multi_output_model(input_dim=100)
    # 18 named outputs
    assert len(model.output_names) == 18
    assert set(model.output_names) == set(LABEL_KEYS)


def test_model_output_shapes():
    """Each head produces the right shape for a single sample."""
    model = build_multi_output_model(input_dim=100)
    x = np.zeros((1, 100), dtype=np.float32)
    preds = model.predict(x, verbose=0)
    # preds is a dict when the model has named outputs
    assert isinstance(preds, dict)
    # 2-class heads
    assert preds["winner"].shape == (1, 2)
    assert preds["winner_kills"].shape == (1, 2)
    # 3-class heads
    for name in ("first_blood", "first_baron", "first_inhibitor", "first_tower"):
        assert preds[name].shape == (1, 3), f"{name} shape mismatch"
    # Binary heads (Dense 1, sigmoid)
    for name in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon"):
        assert preds[name].shape == (1, 1), f"{name} shape mismatch"
    # Regression heads
    for name in ("kill_handicap", "total_kills", "team_a_kills", "team_b_kills",
                 "total_barons", "total_dragons", "total_towers"):
        assert preds[name].shape == (1, 1), f"{name} shape mismatch"


def test_model_trains_one_step():
    """A single train_on_batch call works end-to-end with dict targets."""
    model = build_multi_output_model(input_dim=50)
    x = np.random.rand(4, 50).astype(np.float32)
    targets = {
        "winner":          np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        "winner_kills":    np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        "first_blood":     np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
        "first_baron":     np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
        "first_inhibitor": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
        "first_tower":     np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
        "kills_odd":       np.array([[1], [0], [1], [0]], dtype=np.float32),
        "both_baron":      np.array([[1], [0], [1], [0]], dtype=np.float32),
        "both_inhibitor":  np.array([[1], [0], [1], [0]], dtype=np.float32),
        "both_dragon":     np.array([[1], [0], [1], [0]], dtype=np.float32),
        "elder_dragon":    np.array([[0], [0], [1], [0]], dtype=np.float32),
        "kill_handicap":   np.array([[5.0], [-3.0], [0.0], [10.0]], dtype=np.float32),
        "total_kills":     np.array([[30.0], [25.0], [40.0], [35.0]], dtype=np.float32),
        "team_a_kills":    np.array([[17.5], [11.0], [20.0], [22.5]], dtype=np.float32),
        "team_b_kills":    np.array([[12.5], [14.0], [20.0], [12.5]], dtype=np.float32),
        "total_barons":    np.array([[2.0], [1.0], [3.0], [1.0]], dtype=np.float32),
        "total_dragons":   np.array([[4.0], [2.0], [5.0], [3.0]], dtype=np.float32),
        "total_towers":    np.array([[12.0], [8.0], [15.0], [10.0]], dtype=np.float32),
    }
    result = model.train_on_batch(x, targets)
    # train_on_batch returns a list with overall loss + per-head losses + per-head metrics
    # For multi-output models with 18 heads + total + metrics, expect a non-empty result
    assert result is not None
