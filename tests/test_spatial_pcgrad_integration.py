import pytest
import tensorflow as tf
from core.equivariant_model import build_equivariant_deepsets_model
from core.pcgrad import create_compiled_pcgrad_step


def test_spatial_model_runs_compiled_pcgrad_step():
    model = build_equivariant_deepsets_model(
        player_feature_dim=46,
        pruned_heads=True,
        optimizer_type="adamw",
        weight_decay=1e-3,
        dropout_rate=0.30,
        spatial_noise_std=0.05,
        label_smoothing=0.05,
    )
    optimizer = model.optimizer
    loss_fns = {
        "winner": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        "kill_handicap": tf.keras.losses.MeanSquaredError(),
    }
    loss_weights = {"winner": 1.0, "kill_handicap": 0.1}
    step_fn = create_compiled_pcgrad_step(model, optimizer, loss_fns, loss_weights, jit_compile=False)

    xa = tf.random.normal((8, 5, 46))
    xb = tf.random.normal((8, 5, 46))
    yb = {
        "winner": tf.one_hot(tf.zeros((8,), dtype=tf.int32), 2),
        "kill_handicap": tf.zeros((8, 1)),
    }
    losses = step_fn({"team_a": xa, "team_b": xb}, yb)
    assert "winner" in losses
    assert "kill_handicap" in losses
    assert float(losses["winner"]) > 0.0
    assert float(losses["kill_handicap"]) >= 0.0
