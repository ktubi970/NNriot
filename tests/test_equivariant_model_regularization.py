import pytest
import tensorflow as tf
import numpy as np
from core.equivariant_model import build_equivariant_deepsets_model


def test_model_builds_with_adamw_and_smoothing():
    model = build_equivariant_deepsets_model(
        player_feature_dim=46,
        optimizer_type="adamw",
        weight_decay=1e-3,
        dropout_rate=0.30,
        label_smoothing=0.05,
        pruned_heads=True,
    )
    assert isinstance(model.optimizer, tf.keras.optimizers.AdamW)
    assert model.count_params() > 0
    # Check that categorical crossentropy loss has label smoothing configured
    winner_loss = model.loss.get("winner")
    assert isinstance(winner_loss, tf.keras.losses.CategoricalCrossentropy)
    assert float(winner_loss.label_smoothing) == pytest.approx(0.05)


def test_spatial_jitter_layer_active_in_training_only():
    model = build_equivariant_deepsets_model(
        player_feature_dim=46,
        spatial_noise_std=0.05,
    )
    xa = tf.ones((2, 5, 46))
    xb = tf.ones((2, 5, 46))
    out_eval1 = model({"team_a": xa, "team_b": xb}, training=False)["winner"]
    out_eval2 = model({"team_a": xa, "team_b": xb}, training=False)["winner"]
    # In eval mode, outputs must be identical (deterministic)
    tf.debugging.assert_near(out_eval1, out_eval2)

    # In training mode, outputs should differ due to jitter
    out_tr1 = model({"team_a": xa, "team_b": xb}, training=True)["winner"]
    out_tr2 = model({"team_a": xa, "team_b": xb}, training=True)["winner"]
    diff = tf.reduce_max(tf.abs(out_tr1 - out_tr2))
    assert float(diff) > 0.0


def test_draft_combat_anti_symmetry_and_side_bias_preserved():
    model = build_equivariant_deepsets_model(
        player_feature_dim=46,
        optimizer_type="adamw",
        spatial_noise_std=0.05,
        label_smoothing=0.05,
        enable_side_bias=True,
        blue_win_prior=0.525,
    )
    # Generate mock inputs
    xa = tf.random.normal((4, 5, 46))
    xb = tf.random.normal((4, 5, 46))

    out_ab = model({"team_a": xa, "team_b": xb}, training=False)
    out_ba = model({"team_a": xb, "team_b": xa}, training=False)

    # Pure draft advantage head must be strictly anti-symmetric: P(B, A) == [P_b, P_a]
    draft_p_ab = out_ab["draft_advantage"]
    draft_p_ba = out_ba["draft_advantage"]
    tf.debugging.assert_near(draft_p_ab[:, 0], draft_p_ba[:, 1], atol=1e-5)
    tf.debugging.assert_near(draft_p_ab[:, 1], draft_p_ba[:, 0], atol=1e-5)

    # Mirror matchup on draft_advantage must be exactly 0.50 / 0.50
    out_mirror = model({"team_a": xa, "team_b": xa}, training=False)
    draft_mirror = out_mirror["draft_advantage"]
    tf.debugging.assert_near(draft_mirror, tf.fill(tf.shape(draft_mirror), 0.5), atol=1e-5)

    # Side bias layer must exist and match probability must show blue side advantage
    assert "winner_side_bias" in [layer.name for layer in model.layers]
    # On mirror matchup, winner probability should reflect blue side prior (> 0.5 for blue / team A)
    # Note winner format is [p_team_a (blue), p_team_b (red)]
    winner_mirror = out_mirror["winner"]
    assert float(tf.reduce_mean(winner_mirror[:, 0])) > 0.50
