import pytest
import numpy as np
import tensorflow as tf
from core.deepsets_layers import SiameseTeamEncoder

def test_permutation_invariance_within_team():
    layer = SiameseTeamEncoder(embed_dim=32, latent_dim=64)
    # Batch of 2 matches, 5 players, 8 features
    team_a = tf.random.normal((2, 5, 8))
    team_b = tf.random.normal((2, 5, 8))
    
    diff1, sum1 = layer((team_a, team_b))
    
    # Permute order of players in team_a (e.g. swap slot 0 and 3)
    perm = [3, 1, 2, 0, 4]
    team_a_perm = tf.gather(team_a, perm, axis=1)
    diff2, sum2 = layer((team_a_perm, team_b))
    
    np.testing.assert_allclose(diff1.numpy(), diff2.numpy(), atol=1e-5)
    np.testing.assert_allclose(sum1.numpy(), sum2.numpy(), atol=1e-5)

def test_siamese_team_swap_antisymmetry():
    layer = SiameseTeamEncoder(embed_dim=32, latent_dim=64)
    team_a = tf.random.normal((2, 5, 8))
    team_b = tf.random.normal((2, 5, 8))
    
    diff_ab, sum_ab = layer((team_a, team_b))
    diff_ba, sum_ba = layer((team_b, team_a))
    
    # T_diff must be strictly anti-symmetric: diff(B, A) == -diff(A, B)
    np.testing.assert_allclose(diff_ba.numpy(), -diff_ab.numpy(), atol=1e-5)
    # T_sum must be strictly symmetric: sum(B, A) == sum(A, B)
    np.testing.assert_allclose(sum_ba.numpy(), sum_ab.numpy(), atol=1e-5)
