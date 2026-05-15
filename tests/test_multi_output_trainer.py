import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from continuous_trainer import _build_targets, sparse_microbatch_generator
from feature_labels import LABEL_KEYS


def _make_record(labels: dict) -> dict:
    """Mimics the record shape returned by database.get_untrained_records."""
    return {"id": 1, "match_id": "M1", "feature_json": {"foo": "bar"}, "labels_json": labels}


def _full_labels(**overrides) -> dict:
    """Default-valid labels dict, overridable for specific tests."""
    base = {
        "winner": 0, "team_b_kill_lead": 0,
        "kill_handicap": 5, "total_kills": 30, "team_a_kills": 17, "team_b_kills": 13,
        "kills_odd": 0,
        "first_blood": 0, "first_baron": 0, "first_inhibitor": 0, "first_tower": 0,
        "total_barons": 1, "total_dragons": 3, "total_towers": 11,
        "both_baron": 0, "both_inhibitor": 0, "both_dragon": 1, "elder_dragon": 0,
    }
    base.update(overrides)
    return base


def test_build_targets_shapes_and_keys():
    """All 18 head names present with correct shapes."""
    records = [_make_record(_full_labels()) for _ in range(3)]
    t = _build_targets(records)
    assert set(t.keys()) == set(LABEL_KEYS), "missing or extra head"
    # softmax-2
    for name in ("winner", "team_b_kill_lead"):
        assert t[name].shape == (3, 2)
    # softmax-3
    for name in ("first_blood", "first_baron", "first_inhibitor", "first_tower"):
        assert t[name].shape == (3, 3)
    # sigmoid-1 + regression
    for name in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon",
                 "kill_handicap", "total_kills", "team_a_kills", "team_b_kills",
                 "total_barons", "total_dragons", "total_towers"):
        assert t[name].shape == (3, 1)


def test_build_targets_one_hot_encoding():
    """Categorical heads are one-hot encoded correctly."""
    r0 = _make_record(_full_labels(winner=0, first_baron=2, team_b_kill_lead=1, first_blood=1))
    t = _build_targets([r0])
    np.testing.assert_array_equal(t["winner"][0], [1, 0])
    np.testing.assert_array_equal(t["team_b_kill_lead"][0], [0, 1])
    np.testing.assert_array_equal(t["first_blood"][0], [0, 1, 0])
    np.testing.assert_array_equal(t["first_baron"][0], [0, 0, 1])


def test_build_targets_regression_passthrough():
    """Regression heads receive the raw float values."""
    r0 = _make_record(_full_labels(total_kills=42, kill_handicap=-7, total_dragons=5))
    t = _build_targets([r0])
    assert t["total_kills"][0, 0] == 42.0
    assert t["kill_handicap"][0, 0] == -7.0
    assert t["total_dragons"][0, 0] == 5.0


def test_sparse_microbatch_generator_with_dict_targets():
    """Generator yields dict batches when given dict y."""
    from scipy.sparse import csr_matrix
    x_sparse = csr_matrix(np.eye(10, 5, dtype=np.float32))
    y_dict = {
        "winner": np.eye(10, 2, dtype=np.float32),
        "total_kills": np.arange(10, dtype=np.float32).reshape(-1, 1),
    }
    batches = list(sparse_microbatch_generator(x_sparse, y_dict, batch_size=3))
    assert len(batches) == 4  # 10 samples / batch_size 3 → 4 batches (last is 1)
    for x_batch, y_batch in batches:
        assert isinstance(y_batch, dict)
        assert set(y_batch.keys()) == {"winner", "total_kills"}
        assert y_batch["winner"].shape[0] == x_batch.shape[0]
        assert y_batch["total_kills"].shape[0] == x_batch.shape[0]
