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

def test_extract_team_tensors_preserves_priors():
    custom_priors = [0.65, 0.72, 3.4, 0.60, 0.85, 4.2]
    sample_feat = {
        "team_a": [
            {"champion": "Ahri", "role": "MIDDLE", "priors": custom_priors},
        ],
        "team_b": [],
    }
    ta, tb = extract_team_tensors(sample_feat)
    # Ahri has role MIDDLE -> slot 2 in ROLE_NAMES (TOP=0, JUNGLE=1, MIDDLE=2)
    np.testing.assert_allclose(ta[2, 2:], custom_priors, atol=1e-5)

def test_extract_team_tensors_canonical_role_slots():
    sample_feat = {
        "team_a": [
            {"champion": "Jinx", "role": "BOT"},       # slot 3 (BOTTOM)
            {"champion": "Thresh", "role": "SUP"},      # slot 4 (UTILITY)
            {"champion": "Garen", "role": "TOP"},       # slot 0
            {"champion": "Zed", "role": "MID"},         # slot 2 (MIDDLE)
            {"champion": "LeeSin", "role": "JUNGLE"},   # slot 1
        ],
        "team_b": [],
    }
    ta, _ = extract_team_tensors(sample_feat)
    assert ta[0, 1] == 0.0  # TOP
    assert ta[1, 1] == 1.0  # JUNGLE
    assert ta[2, 1] == 2.0  # MIDDLE
    assert ta[3, 1] == 3.0  # BOTTOM
    assert ta[4, 1] == 4.0  # UTILITY

