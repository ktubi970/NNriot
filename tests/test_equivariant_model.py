import pytest
import numpy as np
import tensorflow as tf
from core.equivariant_model import build_equivariant_deepsets_model

def test_winner_probability_inversion_under_team_swap():
    model = build_equivariant_deepsets_model(player_feature_dim=8, embed_dim=16, latent_dim=32)
    
    # 2 matches, 5 players, 8 features per player
    team_a = tf.constant(np.random.normal(size=(2, 5, 8)), dtype=tf.float32)
    team_b = tf.constant(np.random.normal(size=(2, 5, 8)), dtype=tf.float32)
    
    pred_ab = model({"team_a": team_a, "team_b": team_b})
    pred_ba = model({"team_a": team_b, "team_b": team_a})
    
    win_ab = pred_ab["winner"].numpy()  # shape (2, 2), [P(team_a_wins), P(team_b_wins)]
    win_ba = pred_ba["winner"].numpy()
    
    # P(Team B wins when B is team_b) == P(Team A wins when B is team_a)
    np.testing.assert_allclose(win_ab[:, 1], win_ba[:, 0], atol=1e-5)
    np.testing.assert_allclose(win_ab[:, 0], win_ba[:, 1], atol=1e-5)

def test_total_kills_invariance_under_team_swap():
    model = build_equivariant_deepsets_model(player_feature_dim=8, embed_dim=16, latent_dim=32)
    
    team_a = tf.constant(np.random.normal(size=(2, 5, 8)), dtype=tf.float32)
    team_b = tf.constant(np.random.normal(size=(2, 5, 8)), dtype=tf.float32)
    
    pred_ab = model({"team_a": team_a, "team_b": team_b})
    pred_ba = model({"team_a": team_b, "team_b": team_a})
    
    # Symmetric head total_kills must be identical
    np.testing.assert_allclose(pred_ab["total_kills"].numpy(), pred_ba["total_kills"].numpy(), atol=1e-5)

def test_all_heads_present_in_model():
    from core.feature_labels import ALL_LABEL_KEYS
    model = build_equivariant_deepsets_model(player_feature_dim=8, embed_dim=16, latent_dim=32)
    assert set(model.output_names) == set(ALL_LABEL_KEYS)
