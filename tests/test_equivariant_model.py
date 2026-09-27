import pytest
import numpy as np
import tensorflow as tf
from core.equivariant_model import (
    build_equivariant_deepsets_model,
    SkewSymmetricBilinearDense,
    EvidentialDirichletSoftmax,
)

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

def test_winner_probability_inversion_under_team_swap():
    model = build_equivariant_deepsets_model(player_feature_dim=8, embed_dim=16, latent_dim=32)
    team_a, team_b = _make_teams()
    
    pred_ab = model({"team_a": team_a, "team_b": team_b})
    pred_ba = model({"team_a": team_b, "team_b": team_a})
    
    win_ab = pred_ab["winner"].numpy()  # shape (2, 2), [P(team_a_wins), P(team_b_wins)]
    win_ba = pred_ba["winner"].numpy()
    
    # P(Team B wins when B is team_b) == P(Team A wins when B is team_a)
    np.testing.assert_allclose(win_ab[:, 1], win_ba[:, 0], atol=1e-5)
    np.testing.assert_allclose(win_ab[:, 0], win_ba[:, 1], atol=1e-5)

def test_total_kills_invariance_under_team_swap():
    model = build_equivariant_deepsets_model(player_feature_dim=8, embed_dim=16, latent_dim=32)
    team_a, team_b = _make_teams()
    
    pred_ab = model({"team_a": team_a, "team_b": team_b})
    pred_ba = model({"team_a": team_b, "team_b": team_a})
    
    # Symmetric head total_kills must be identical
    np.testing.assert_allclose(pred_ab["total_kills"].numpy(), pred_ba["total_kills"].numpy(), atol=1e-5)

def test_all_heads_present_in_model():
    from core.feature_labels import ALL_LABEL_KEYS
    model = build_equivariant_deepsets_model(
        player_feature_dim=8, embed_dim=16, latent_dim=32, pruned_heads=False, use_evidential=False
    )
    assert set(model.output_names) == set(ALL_LABEL_KEYS)


def test_winner_probability_inversion_enriched_14dim():
    from core.features.pipeline import ENRICHED_PLAYER_TENSOR_DIM
    rng = np.random.default_rng(123)
    t_a = np.zeros((2, 5, ENRICHED_PLAYER_TENSOR_DIM), dtype=np.float32)
    t_b = np.zeros((2, 5, ENRICHED_PLAYER_TENSOR_DIM), dtype=np.float32)
    for b in range(2):
        for slot in range(5):
            t_a[b, slot, 0] = float(rng.integers(1, 150))
            t_a[b, slot, 1] = float(slot)
            t_a[b, slot, 2:] = rng.uniform(0.1, 0.9, size=ENRICHED_PLAYER_TENSOR_DIM - 2)

            t_b[b, slot, 0] = float(rng.integers(1, 150))
            t_b[b, slot, 1] = float(slot)
            t_b[b, slot, 2:] = rng.uniform(0.1, 0.9, size=ENRICHED_PLAYER_TENSOR_DIM - 2)

    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_TENSOR_DIM, embed_dim=16, latent_dim=32, use_cross_attention=True
    )
    pred_ab = model({"team_a": tf.constant(t_a), "team_b": tf.constant(t_b)})
    pred_ba = model({"team_a": tf.constant(t_b), "team_b": tf.constant(t_a)})

    win_ab = pred_ab["winner"].numpy()
    win_ba = pred_ba["winner"].numpy()

    np.testing.assert_allclose(win_ab[:, 1], win_ba[:, 0], atol=1e-5)
    np.testing.assert_allclose(win_ab[:, 0], win_ba[:, 1], atol=1e-5)


def test_winner_probability_inversion_with_macro_clash():
    from core.features.pipeline import ENRICHED_PLAYER_TENSOR_DIM
    from core.features.draft_composition import MACRO_CLASH_DIM
    rng = np.random.default_rng(456)
    t_a = np.zeros((2, 5, ENRICHED_PLAYER_TENSOR_DIM), dtype=np.float32)
    t_b = np.zeros((2, 5, ENRICHED_PLAYER_TENSOR_DIM), dtype=np.float32)
    macro_ab = rng.uniform(-0.5, 0.5, size=(2, MACRO_CLASH_DIM)).astype(np.float32)
    macro_ba = -macro_ab  # Anti-symmetric swap

    for b in range(2):
        for slot in range(5):
            t_a[b, slot, 0] = float(rng.integers(1, 150))
            t_a[b, slot, 1] = float(slot)
            t_a[b, slot, 2:] = rng.uniform(0.1, 0.9, size=ENRICHED_PLAYER_TENSOR_DIM - 2)

            t_b[b, slot, 0] = float(rng.integers(1, 150))
            t_b[b, slot, 1] = float(slot)
            t_b[b, slot, 2:] = rng.uniform(0.1, 0.9, size=ENRICHED_PLAYER_TENSOR_DIM - 2)

    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_TENSOR_DIM,
        macro_feature_dim=MACRO_CLASH_DIM,
        embed_dim=16,
        latent_dim=32,
        include_macro_clash=True,
    )
    pred_ab = model({"team_a": tf.constant(t_a), "team_b": tf.constant(t_b), "macro_clash": tf.constant(macro_ab)})
    pred_ba = model({"team_a": tf.constant(t_b), "team_b": tf.constant(t_a), "macro_clash": tf.constant(macro_ba)})

    win_ab = pred_ab["winner"].numpy()
    win_ba = pred_ba["winner"].numpy()

    # Exact anti-symmetry guarantee: P(B|A, macro_ab) == 1 - P(B|B, -macro_ab)
    np.testing.assert_allclose(win_ab[:, 1], win_ba[:, 0], atol=1e-5)
    np.testing.assert_allclose(win_ab[:, 0], win_ba[:, 1], atol=1e-5)


def test_skew_symmetric_bilinear_dense():
    layer = SkewSymmetricBilinearDense()
    t_a = tf.random.normal((4, 32))
    t_b = tf.random.normal((4, 32))
    s_ab = layer((t_a, t_b))
    s_ba = layer((t_b, t_a))

    assert s_ab.shape == (4, 1)
    # Anti-symmetry: s(B, A) == -s(A, B)
    np.testing.assert_allclose(s_ab.numpy(), -s_ba.numpy(), atol=1e-5)

    # Self-matchup symmetry: s(A, A) == 0
    s_aa = layer((t_a, t_a))
    np.testing.assert_allclose(s_aa.numpy(), 0.0, atol=1e-5)

    # Positional and list calls
    s_pos = layer(t_a, t_b)
    s_list = layer([t_a, t_b])
    np.testing.assert_allclose(s_pos.numpy(), s_ab.numpy(), atol=1e-6)
    np.testing.assert_allclose(s_list.numpy(), s_ab.numpy(), atol=1e-6)

    # Serialization roundtrip
    cfg = layer.get_config()
    restored = SkewSymmetricBilinearDense.from_config(cfg)
    assert restored is not None


def test_evidential_dirichlet_head():
    layer = EvidentialDirichletSoftmax()
    s = tf.constant([[2.5], [-1.2], [0.0], [4.0]], dtype=tf.float32)
    p_ab, u_ab = layer(s)
    p_ba, u_ba = layer(-s)

    # Shape checks
    assert p_ab.shape == (4, 2)
    assert u_ab.shape == (4, 1)

    # Probabilities sum to 1
    np.testing.assert_allclose(tf.reduce_sum(p_ab, axis=-1).numpy(), 1.0, atol=1e-5)

    # Anti-symmetry: P(Team B wins | s) == P(Team A wins | -s) == 1 - P(Team B wins | -s)
    np.testing.assert_allclose(p_ab.numpy()[:, 1], p_ba.numpy()[:, 0], atol=1e-5)
    np.testing.assert_allclose(p_ab.numpy()[:, 0], p_ba.numpy()[:, 1], atol=1e-5)
    np.testing.assert_allclose(p_ab.numpy()[:, 1], 1.0 - p_ba.numpy()[:, 1], atol=1e-5)

    # Uncertainty invariance: u(B, A) == u(A, B)
    np.testing.assert_allclose(u_ab.numpy(), u_ba.numpy(), atol=1e-5)

    # Epistemic uncertainty bounds: 0 < u <= 1
    assert np.all(u_ab.numpy() > 0.0)
    assert np.all(u_ab.numpy() <= 1.0)

    # Dict output mode
    layer_dict = EvidentialDirichletSoftmax(return_dict=True)
    out_dict = layer_dict(s)
    assert "winner" in out_dict
    assert "winner_uncertainty" in out_dict
    np.testing.assert_allclose(out_dict["winner"].numpy(), p_ab.numpy(), atol=1e-6)
    np.testing.assert_allclose(out_dict["winner_uncertainty"].numpy(), u_ab.numpy(), atol=1e-6)

    # Serialization
    cfg = layer.get_config()
    restored = EvidentialDirichletSoftmax.from_config(cfg)
    assert restored is not None


def test_build_equivariant_model_pruned_heads():
    from core.feature_labels import PRUNED_LABEL_KEYS
    model = build_equivariant_deepsets_model(
        player_feature_dim=14,
        pruned_heads=True,
        use_evidential=True,
    )
    expected_heads = set(PRUNED_LABEL_KEYS) | {"winner_uncertainty"}
    assert set(model.output_names) == expected_heads

    team_a = tf.random.normal((2, 5, 14))
    team_b = tf.random.normal((2, 5, 14))

    pred_ab = model({"team_a": team_a, "team_b": team_b})
    pred_ba = model({"team_a": team_b, "team_b": team_a})

    # 1. Winner probability anti-symmetry
    win_ab = pred_ab["winner"].numpy()
    win_ba = pred_ba["winner"].numpy()
    np.testing.assert_allclose(win_ab[:, 1], 1.0 - win_ba[:, 1], atol=1e-5)
    np.testing.assert_allclose(win_ab[:, 0], win_ba[:, 1], atol=1e-5)

    # 2. Winner uncertainty invariance & bounds
    u_ab = pred_ab["winner_uncertainty"].numpy()
    u_ba = pred_ba["winner_uncertainty"].numpy()
    np.testing.assert_allclose(u_ab, u_ba, atol=1e-5)
    assert np.all(u_ab > 0.0)
    assert np.all(u_ab <= 1.0)

    # 3. Kill handicap anti-symmetry
    np.testing.assert_allclose(
        pred_ab["kill_handicap"].numpy(),
        -pred_ba["kill_handicap"].numpy(),
        atol=1e-5,
    )

    # 4. Team kills swap
    np.testing.assert_allclose(
        pred_ab["team_a_kills"].numpy(),
        pred_ba["team_b_kills"].numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pred_ab["team_b_kills"].numpy(),
        pred_ba["team_a_kills"].numpy(),
        atol=1e-5,
    )

    # 5. Total kills symmetry
    np.testing.assert_allclose(
        pred_ab["total_kills"].numpy(),
        pred_ba["total_kills"].numpy(),
        atol=1e-5,
    )

    # 6. Team B kill lead anti-symmetry
    np.testing.assert_allclose(
        pred_ab["team_b_kill_lead"].numpy()[:, 1],
        1.0 - pred_ba["team_b_kill_lead"].numpy()[:, 1],
        atol=1e-5,
    )

    # 7. First blood equivariant swap (class 0=Team A, class 1=Team B, class 2=Neither)
    fb_ab = pred_ab["first_blood"].numpy()
    fb_ba = pred_ba["first_blood"].numpy()
    np.testing.assert_allclose(fb_ab[:, 0], fb_ba[:, 1], atol=1e-5)
    np.testing.assert_allclose(fb_ab[:, 1], fb_ba[:, 0], atol=1e-5)
    np.testing.assert_allclose(fb_ab[:, 2], fb_ba[:, 2], atol=1e-5)



