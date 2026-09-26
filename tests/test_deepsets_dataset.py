import pytest
import numpy as np
from core.deepsets_dataset import extract_team_tensors, batch_extract_team_tensors

def test_extract_team_tensors_shape_and_types():
    sample_feat = {
        "team_a": [
            {"champion": "Ahri", "role": "MIDDLE"},
            {"champion": "LeeSin", "role": "JUNGLE"},
            {"champion": "Garen", "role": "TOP"},
            {"champion": "Jinx", "role": "BOTTOM"},
            {"champion": "Thresh", "role": "UTILITY"},
        ],
        "team_b": [
            {"champion": "Zed", "role": "MIDDLE"},
            {"champion": "Elise", "role": "JUNGLE"},
            {"champion": "Darius", "role": "TOP"},
            {"champion": "Vayne", "role": "BOTTOM"},
            {"champion": "Lulu", "role": "UTILITY"},
        ],
    }
    ta, tb = extract_team_tensors(sample_feat)
    assert ta.shape == (5, 8)
    assert tb.shape == (5, 8)
    assert ta.dtype == np.float32
    assert tb.dtype == np.float32

def test_extract_team_tensors_missing_or_empty():
    ta, tb = extract_team_tensors(None)
    assert ta.shape == (5, 8)
    assert tb.shape == (5, 8)

def test_batch_extract_team_tensors():
    sample_feat = {
        "team_a": [{"champion": "Ahri", "role": "MIDDLE"}] * 5,
        "team_b": [{"champion": "Zed", "role": "MIDDLE"}] * 5,
    }
    batch = batch_extract_team_tensors([sample_feat, sample_feat])
    assert "team_a" in batch
    assert "team_b" in batch
    assert batch["team_a"].shape == (2, 5, 8)
    assert batch["team_b"].shape == (2, 5, 8)
