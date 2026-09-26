import pytest
import numpy as np
import tensorflow as tf
from core.deepsets_layers import SiameseTeamEncoder

def _make_teams():
    rng = np.random.default_rng(42)
    t_a = np.zeros((2, 5, 8), dtype=np.float32)
    t_b = np.zeros((2, 5, 8), dtype=np.float32)
    for b in range(2):
        for slot in range(5):
            t_a[b, slot, 0] = float(rng.integers(1, 150))
            t_a[b, slot, 1] = float(slot)
            t_a[b, slot, 2:] = rng.uniform(0.2, 0.8, size=6)

            t_b[b, slot, 0] = float(rng.integers(1, 150))
            t_b[b, slot, 1] = float(slot)
            t_b[b, slot, 2:] = rng.uniform(0.2, 0.8, size=6)
    return tf.constant(t_a), tf.constant(t_b)

def test_permutation_invariance_within_team():
    layer = SiameseTeamEncoder(embed_dim=32, latent_dim=64)
    team_a, team_b = _make_teams()
    
    diff1, sum1, ta1, tb1 = layer((team_a, team_b))
    
    # Permute order of players in team_a (e.g. swap slot 0 and 3)
    perm = [3, 1, 2, 0, 4]
    team_a_perm = tf.gather(team_a, perm, axis=1)
    diff2, sum2, ta2, tb2 = layer((team_a_perm, team_b))
    
    np.testing.assert_allclose(diff1.numpy(), diff2.numpy(), atol=1e-5)
    np.testing.assert_allclose(sum1.numpy(), sum2.numpy(), atol=1e-5)
    np.testing.assert_allclose(ta1.numpy(), ta2.numpy(), atol=1e-5)

def test_siamese_team_swap_antisymmetry():
    layer = SiameseTeamEncoder(embed_dim=32, latent_dim=64)
    team_a, team_b = _make_teams()
    
    diff_ab, sum_ab, ta_ab, tb_ab = layer((team_a, team_b))
    diff_ba, sum_ba, ta_ba, tb_ba = layer((team_b, team_a))
    
    # T_diff must be strictly anti-symmetric: diff(B, A) == -diff(A, B)
    np.testing.assert_allclose(diff_ba.numpy(), -diff_ab.numpy(), atol=1e-5)
    # T_sum must be strictly symmetric: sum(B, A) == sum(A, B)
    np.testing.assert_allclose(sum_ba.numpy(), sum_ab.numpy(), atol=1e-5)
    # Team A and B representations swap
    np.testing.assert_allclose(ta_ba.numpy(), tb_ab.numpy(), atol=1e-5)
    np.testing.assert_allclose(tb_ba.numpy(), ta_ab.numpy(), atol=1e-5)

