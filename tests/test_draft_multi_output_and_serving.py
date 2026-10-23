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
    from core.serving_contract import ContractError
    import core.prediction_service as serving
    def failed(**kwargs):raise ContractError('Unsupported legacy dense feature contract')
    monkeypatch.setattr(serving,'get_prediction_service',failed)
    legacy=MagicMock();monkeypatch.setattr(app_core,'global_trainer',legacy)
    result=final_predict_match_outcome({'participants':[],'teams':[]})
    assert result['error_code']=='model_contract'
    legacy.predict.assert_not_called()



def test_final_predict_uses_verified_serving_contract(monkeypatch, tmp_path):
    from tests.serving_fixture import install_serving_fixture,full_match
    install_serving_fixture(monkeypatch,{'winner':np.array([[.7,.3]])}, tmp_path)
    result=final_predict_match_outcome(full_match())
    assert result['success']
    assert result['winner']['predicted']=='team_a'
    assert result['model']['feature_contract']=='nnriot.players14.v1'
    assert result['heads_raw']['winner']==[.7,.3]
