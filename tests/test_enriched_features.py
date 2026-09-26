"""Unit tests for enriched features module."""
import numpy as np
import pytest

from core.enriched_features import (
    ENRICHED_DENSE_DIM,
    ENRICHED_TABULAR_DIM,
    extract_enriched_features,
    batch_extract_enriched_features,
)


def test_extract_enriched_features_dense_dimension():
    feature_json = {
        "team_a": [
            {"champion": "Shen", "role": "TOP"},
            {"champion": "LeeSin", "role": "JUNGLE"},
            {"champion": "Sylas", "role": "MIDDLE"},
            {"champion": "Tristana", "role": "BOTTOM"},
            {"champion": "Braum", "role": "UTILITY"},
        ],
        "team_b": [
            {"champion": "Neeko", "role": "TOP"},
            {"champion": "XinZhao", "role": "JUNGLE"},
            {"champion": "Zed", "role": "MIDDLE"},
            {"champion": "Ezreal", "role": "BOTTOM"},
            {"champion": "Pyke", "role": "UTILITY"},
        ],
    }

    dense_vec = extract_enriched_features(feature_json, mode="dense")
    assert dense_vec.shape == (ENRICHED_DENSE_DIM,)
    assert dense_vec.dtype == np.float32

    tabular_vec = extract_enriched_features(feature_json, mode="tabular")
    assert tabular_vec.shape == (ENRICHED_TABULAR_DIM,)
    assert tabular_vec.dtype == np.float32


def test_batch_extract_enriched_features():
    feature_json = {
        "team_a": [{"champion": "Shen", "role": "TOP"} for _ in range(5)],
        "team_b": [{"champion": "Zed", "role": "MIDDLE"} for _ in range(5)],
    }
    batch = [feature_json, feature_json, None]
    dense_batch = batch_extract_enriched_features(batch, mode="dense")
    assert dense_batch.shape == (3, ENRICHED_DENSE_DIM)

    empty_batch = batch_extract_enriched_features([], mode="dense")
    assert empty_batch.shape == (0, ENRICHED_DENSE_DIM)


def test_extract_enriched_features_invalid_mode():
    with pytest.raises(ValueError, match="mode must be"):
        extract_enriched_features({}, mode="invalid")
