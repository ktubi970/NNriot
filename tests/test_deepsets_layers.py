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


def test_siamese_team_encoder_pma_and_sinkhorn():
    encoder = SiameseTeamEncoder(
        embed_dim=32,
        latent_dim=64,
        use_set_transformer=True,
        use_sinkhorn_attention=True,
        use_periodic_embeddings=True,
    )
    team_a = tf.random.normal((2, 5, 14))
    team_b = tf.random.normal((2, 5, 14))

    t_diff_ab, t_sum_ab, ta_ab, tb_ab = encoder((team_a, team_b))
    t_diff_ba, t_sum_ba, ta_ba, tb_ba = encoder((team_b, team_a))

    # Anti-symmetry and symmetry
    np.testing.assert_allclose(t_diff_ba.numpy(), -t_diff_ab.numpy(), atol=1e-5)
    np.testing.assert_allclose(t_sum_ba.numpy(), t_sum_ab.numpy(), atol=1e-5)
    # Team cross-consistency
    np.testing.assert_allclose(ta_ba.numpy(), tb_ab.numpy(), atol=1e-5)
    np.testing.assert_allclose(tb_ba.numpy(), ta_ab.numpy(), atol=1e-5)


def test_sinkhorn_cross_attention_doubly_stochastic():
    from core.deepsets_layers import SinkhornCrossAttention
    layer = SinkhornCrossAttention(embed_dim=32, num_iterations=10)
    team_a = tf.random.normal((3, 5, 32))
    team_b = tf.random.normal((3, 5, 32))

    plan = layer.compute_transport_plan(team_a, team_b)
    assert plan.shape == (3, 5, 5)

    # Doubly stochastic: row sums = 1.0, col sums = 1.0
    row_sums = tf.reduce_sum(plan, axis=-1).numpy()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
    col_sums = tf.reduce_sum(plan, axis=-2).numpy()
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-5)
    assert np.all(plan.numpy() >= 0.0)

    # Swap transposes the transport plan
    plan_swap = layer.compute_transport_plan(team_b, team_a)
    np.testing.assert_allclose(plan_swap.numpy(), tf.transpose(plan, [0, 2, 1]).numpy(), atol=1e-5)

    # Test output anti-symmetry and symmetry
    out_a, out_b = layer((team_a, team_b))
    out_a_swap, out_b_swap = layer((team_b, team_a))
    np.testing.assert_allclose(out_a_swap.numpy(), out_b.numpy(), atol=1e-5)
    np.testing.assert_allclose(out_b_swap.numpy(), out_a.numpy(), atol=1e-5)


def test_set_transformer_pma_permutation_invariance():
    from core.deepsets_layers import SetTransformerPMA
    layer = SetTransformerPMA(embed_dim=32, num_heads=2, num_seeds=2)
    x = tf.random.normal((2, 5, 32))
    perm = [3, 1, 2, 0, 4]
    x_perm = tf.gather(x, perm, axis=1)

    out1 = layer(x)
    out2 = layer(x_perm)

    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-5)


def test_layers_serialization():
    from core.deepsets_layers import SetTransformerPMA, SinkhornCrossAttention, SiameseTeamEncoder

    # 1. SetTransformerPMA serialization
    pma = SetTransformerPMA(embed_dim=16, num_heads=2, num_seeds=2)
    pma_config = pma.get_config()
    pma_restored = SetTransformerPMA.from_config(pma_config)
    assert pma_restored.embed_dim == 16
    assert pma_restored.num_heads == 2
    assert pma_restored.num_seeds == 2
    # Forward pass on restored PMA
    pma_out = pma_restored(tf.random.normal((2, 5, 16)))
    assert pma_out.shape == (2, 2 * 16)

    # 2. SinkhornCrossAttention serialization
    sinkhorn = SinkhornCrossAttention(embed_dim=16, temperature=1.0, num_iterations=10)
    sinkhorn_config = sinkhorn.get_config()
    sinkhorn_restored = SinkhornCrossAttention.from_config(sinkhorn_config)
    assert sinkhorn_restored.embed_dim == 16
    assert sinkhorn_restored.temperature == 1.0
    assert sinkhorn_restored.num_iterations == 10
    # Forward pass on restored Sinkhorn
    out_a, out_b = sinkhorn_restored((tf.random.normal((2, 5, 16)), tf.random.normal((2, 5, 16))))
    assert out_a.shape == (2, 5, 16)
    assert out_b.shape == (2, 5, 16)

    # 3. SiameseTeamEncoder serialization
    encoder = SiameseTeamEncoder(
        embed_dim=16,
        latent_dim=32,
        use_set_transformer=True,
        use_sinkhorn_attention=True,
        use_periodic_embeddings=True,
    )
    encoder_config = encoder.get_config()
    encoder_restored = SiameseTeamEncoder.from_config(encoder_config)
    assert encoder_restored.use_set_transformer is True
    assert encoder_restored.use_sinkhorn_attention is True
    assert encoder_restored.use_periodic_embeddings is True
    # Forward pass on restored SiameseTeamEncoder
    t_diff, t_sum, ta, tb = encoder_restored((tf.random.normal((2, 5, 14)), tf.random.normal((2, 5, 14))))
    assert t_diff.shape[0] == 2
    assert t_sum.shape[0] == 2
    assert ta.shape[0] == 2
    assert tb.shape[0] == 2

