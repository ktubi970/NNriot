"""Test spatial trajectory pipeline integration and equivariant model anti-symmetry with spatial features."""
import numpy as np
import pytest
import tensorflow as tf

from core.features.pipeline import (
    DraftFeaturePipeline,
    ENRICHED_PLAYER_TENSOR_DIM,
    ENRICHED_PLAYER_SPATIAL_TENSOR_DIM,
    SPATIAL_TRAJECTORY_DIM,
)
from core.player_priors import PlayerPriorsTracker
from core.equivariant_model import build_equivariant_deepsets_model


def test_pipeline_spatial_dimensions_and_defaults():
    tracker = PlayerPriorsTracker()
    pipeline = DraftFeaturePipeline(player_tracker=tracker)

    # Mock match draft
    feature_dict = {
        "team_a": [
            {"puuid": "P_TOP_A", "role": "TOP", "champion": "Aatrox"},
            {"puuid": "P_JGL_A", "role": "JUNGLE", "champion": "LeeSin"},
            {"puuid": "P_MID_A", "role": "MIDDLE", "champion": "Ahri"},
            {"puuid": "P_BOT_A", "role": "BOTTOM", "champion": "Jinx"},
            {"puuid": "P_SUP_A", "role": "UTILITY", "champion": "Thresh"},
        ],
        "team_b": [
            {"puuid": "P_TOP_B", "role": "TOP", "champion": "Fiora"},
            {"puuid": "P_JGL_B", "role": "JUNGLE", "champion": "Vi"},
            {"puuid": "P_MID_B", "role": "MIDDLE", "champion": "Syndra"},
            {"puuid": "P_BOT_B", "role": "BOTTOM", "champion": "Kaisa"},
            {"puuid": "P_SUP_B", "role": "UTILITY", "champion": "Nautilus"},
        ],
    }

    # Baseline extraction (14 dims)
    t_a_base, t_b_base = pipeline.extract_tensors(feature_dict, include_spatial=False)
    assert t_a_base.shape == (5, ENRICHED_PLAYER_TENSOR_DIM)
    assert t_b_base.shape == (5, ENRICHED_PLAYER_TENSOR_DIM)

    # Spatial extraction without player history (46 dims, zero-padded spatial slice)
    t_a_spat, t_b_spat = pipeline.extract_tensors(feature_dict, include_spatial=True)
    assert t_a_spat.shape == (5, ENRICHED_PLAYER_SPATIAL_TENSOR_DIM)
    assert t_b_spat.shape == (5, ENRICHED_PLAYER_SPATIAL_TENSOR_DIM)
    assert ENRICHED_PLAYER_SPATIAL_TENSOR_DIM == 46

    # Verify baseline slice matches
    np.testing.assert_allclose(t_a_spat[:, :14], t_a_base)
    np.testing.assert_allclose(t_b_spat[:, :14], t_b_base)
    # Verify spatial slice is zero when no history recorded
    assert np.all(t_a_spat[:, 14:46] == 0.0)


def test_pipeline_with_recorded_player_spatial_embeddings():
    tracker = PlayerPriorsTracker()
    # Add dummy 32D spatial embedding for player P_MID_A
    dummy_spatial_emb = np.linspace(0.1, 0.8, 32, dtype=np.float32)
    tracker.record_spatial_embedding("P_MID_A", dummy_spatial_emb)

    pipeline = DraftFeaturePipeline(player_tracker=tracker)

    feature_dict = {
        "team_a": [
            {"puuid": "P_TOP_A", "role": "TOP", "champion": "Aatrox"},
            {"puuid": "P_JGL_A", "role": "JUNGLE", "champion": "LeeSin"},
            {"puuid": "P_MID_A", "role": "MIDDLE", "champion": "Ahri"},
            {"puuid": "P_BOT_A", "role": "BOTTOM", "champion": "Jinx"},
            {"puuid": "P_SUP_A", "role": "UTILITY", "champion": "Thresh"},
        ],
        "team_b": [
            {"puuid": "P_TOP_B", "role": "TOP", "champion": "Fiora"},
            {"puuid": "P_JGL_B", "role": "JUNGLE", "champion": "Vi"},
            {"puuid": "P_MID_B", "role": "MIDDLE", "champion": "Syndra"},
            {"puuid": "P_BOT_B", "role": "BOTTOM", "champion": "Kaisa"},
            {"puuid": "P_SUP_B", "role": "UTILITY", "champion": "Nautilus"},
        ],
    }

    t_a_spat, _ = pipeline.extract_tensors(feature_dict, include_spatial=True)

    # Slot 2 is MIDDLE
    np.testing.assert_allclose(t_a_spat[2, 14:46], dummy_spatial_emb, rtol=1e-5)


def test_equivariant_deepsets_spatial_anti_symmetry():
    # Build model accepting 46-dimensional player features
    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_SPATIAL_TENSOR_DIM,
        use_evidential=True,
        pruned_heads=True,
    )

    batch_size = 2
    team_a = np.random.uniform(0.0, 1.0, size=(batch_size, 5, 46)).astype(np.float32)
    team_b = np.random.uniform(0.0, 1.0, size=(batch_size, 5, 46)).astype(np.float32)

    pred_ab = model({"team_a": team_a, "team_b": team_b})
    pred_ba = model({"team_a": team_b, "team_b": team_a})

    # Draft advantage head guarantees strict anti-symmetry per AGENTS.md convention 1:
    # P(draft_B | B, A) = 1 - P(draft_B | A, B)
    draft_ab = pred_ab["draft_advantage"].numpy()[:, 1]
    draft_ba = pred_ba["draft_advantage"].numpy()[:, 1]
    np.testing.assert_allclose(draft_ab + draft_ba, 1.0, atol=1e-5)

    # When side bias is disabled, winner head also preserves exact mathematical anti-symmetry
    neutral_model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_SPATIAL_TENSOR_DIM,
        use_evidential=True,
        pruned_heads=True,
        enable_side_bias=False,
    )
    n_pred_ab = neutral_model({"team_a": team_a, "team_b": team_b})
    n_pred_ba = neutral_model({"team_a": team_b, "team_b": team_a})
    n_win_ab = n_pred_ab["winner"].numpy()[:, 1]
    n_win_ba = n_pred_ba["winner"].numpy()[:, 1]
    np.testing.assert_allclose(n_win_ab + n_win_ba, 1.0, atol=1e-5)

