"""Tests for compact draft multi-output architecture and dynamic serving."""
from unittest.mock import MagicMock
import numpy as np
import pytest

from core.draft_features import DENSE_FEATURE_DIM
from core.generate_graph import build_draft_multi_output_model
from core.feature_labels import ALL_LABEL_KEYS
from app.core import _format_multi_output_response, final_predict_match_outcome
from app import core as app_core


def test_build_draft_multi_output_model_structure():
    model = build_draft_multi_output_model()
    assert model.input_shape[-1] == DENSE_FEATURE_DIM
    assert set(model.output_names) == set(ALL_LABEL_KEYS)
    # Check compact parameter count (< 600,000 params vs 25M)
    assert model.count_params() < 600_000

    # Sanity forward pass
    dummy = np.zeros((2, DENSE_FEATURE_DIM), dtype=np.float32)
    preds = model(dummy)
    assert "winner" in preds
    assert preds["winner"].shape == (2, 2)


def test_format_multi_output_response_with_single_head_winner():
    # Only "winner" head present in preds
    single_head_preds = {
        "winner": np.array([[0.25, 0.75]], dtype=np.float32),
    }
    resp = _format_multi_output_response(single_head_preds)
    assert resp["success"] is True
    assert resp["winner"]["predicted"] == "B"
    assert resp["winner"]["confidence"] == 0.75
    # Secondary heads are populated with clean fallback defaults
    assert "kills" in resp
    assert "first" in resp
    assert "totals" in resp
    assert "both_teams" in resp


def test_final_predict_dynamically_handles_dense_draft_shape(monkeypatch):
    fake_trainer = MagicMock()
    fake_model = MagicMock()
    fake_model.input_shape = (None, DENSE_FEATURE_DIM)
    fake_model.is_calibrated = True
    fake_model.calibration_temperature = 1.0

    fake_trainer.model = fake_model
    fake_trainer.is_calibrated = True
    fake_trainer.predict.return_value = {
        "winner": np.array([[0.35, 0.65]], dtype=np.float32),
    }

    monkeypatch.setattr(app_core, "global_trainer", fake_trainer)
    monkeypatch.setattr(app_core, "tf_available", True)

    sample_match = {
        "participants": [
            {"champion": "Ahri", "role": "MIDDLE", "team_id": 100},
            {"champion": "Zed", "role": "MIDDLE", "team_id": 200},
        ],
        "teams": [{"team_id": 100}, {"team_id": 200}],
    }
    result = final_predict_match_outcome(sample_match)
    assert "error" not in result
    assert result["success"] is True
    assert result["winner"]["predicted"] == "B"
    # Ensure predict was called with a dense vector of shape (1, DENSE_FEATURE_DIM)
    call_arg = fake_trainer.predict.call_args[0][0]
    assert call_arg.shape == (1, DENSE_FEATURE_DIM)


def test_final_predict_dynamically_handles_enriched_shape(monkeypatch):
    from core.enriched_features import ENRICHED_DENSE_DIM
    fake_trainer = MagicMock()
    fake_model = MagicMock()
    fake_model.input_shape = (None, ENRICHED_DENSE_DIM)
    fake_model.is_calibrated = True
    fake_model.calibration_temperature = 1.0

    fake_trainer.model = fake_model
    fake_trainer.is_calibrated = True
    fake_trainer.predict.return_value = {
        "winner": np.array([[0.70, 0.30]], dtype=np.float32),
    }

    monkeypatch.setattr(app_core, "global_trainer", fake_trainer)
    monkeypatch.setattr(app_core, "tf_available", True)

    sample_match = {
        "participants": [
            {"champion": "Shen", "role": "TOP", "team_id": 100},
            {"champion": "Zed", "role": "MIDDLE", "team_id": 200},
        ],
        "teams": [{"team_id": 100}, {"team_id": 200}],
    }
    result = final_predict_match_outcome(sample_match)
    assert "error" not in result
    assert result["success"] is True
    assert result["winner"]["predicted"] == "A"
    call_arg = fake_trainer.predict.call_args[0][0]
    assert call_arg.shape == (1, ENRICHED_DENSE_DIM)

