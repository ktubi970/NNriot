import pytest
import tensorflow as tf
import numpy as np

from core.generate_graph import (
    HomoscedasticLossLayer,
    PRUNED_LABEL_KEYS,
    PRUNED_LOSS_PER_HEAD,
    PRUNED_LOSS_WEIGHTS,
    PRUNED_METRICS_PER_HEAD,
)
import core.feature_labels as fl


def test_pruned_label_keys_structure():
    expected_keys = (
        "winner",
        "team_b_kill_lead",
        "kill_handicap",
        "first_blood",
        "total_kills",
        "team_a_kills",
        "team_b_kills",
    )
    assert PRUNED_LABEL_KEYS == expected_keys
    assert fl.PRUNED_LABEL_KEYS == expected_keys
    assert len(PRUNED_LABEL_KEYS) == 7
    # All pruned keys must be part of ALL_LABEL_KEYS
    for k in PRUNED_LABEL_KEYS:
        assert k in fl.ALL_LABEL_KEYS

    # Verify dictionaries in generate_graph
    assert set(PRUNED_LOSS_PER_HEAD.keys()) == set(PRUNED_LABEL_KEYS)
    assert set(PRUNED_LOSS_WEIGHTS.keys()) == set(PRUNED_LABEL_KEYS)
    assert set(PRUNED_METRICS_PER_HEAD.keys()) == set(PRUNED_LABEL_KEYS)


def test_homoscedastic_loss_layer_forward():
    layer = HomoscedasticLossLayer(task_names=list(PRUNED_LABEL_KEYS))
    assert len(PRUNED_LABEL_KEYS) == 7

    # 7 dummy losses each equal to 1.0
    losses = {k: tf.constant(1.0, dtype=tf.float32) for k in PRUNED_LABEL_KEYS}
    total_loss = layer(losses)

    assert total_loss.shape == ()
    assert not tf.math.is_nan(total_loss)
    assert len(layer.trainable_weights) == 1
    # Weight shape should be (7,) representing log_vars (s_t)
    assert layer.trainable_weights[0].shape == (7,)

    # With initial log_vars = 0.0, exp(-0) * 1.0 + 0.5 * 0 = 1.0 per task.
    # Total for 7 tasks should be exactly 7.0.
    np.testing.assert_allclose(total_loss.numpy(), 7.0, atol=1e-5)


def test_homoscedastic_loss_gradient_flow_model_and_weights():
    # Verify gradient flow to BOTH model parameters and layer log_vars
    task_names = list(PRUNED_LABEL_KEYS)
    layer = HomoscedasticLossLayer(task_names=task_names)

    # Toy model producing predictions for each task
    model_var = tf.Variable(tf.ones((4, len(task_names)), dtype=tf.float32), trainable=True)
    targets = tf.zeros((4, len(task_names)), dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        # Each task prediction is model_var[:, i]
        task_losses = {}
        for idx, name in enumerate(task_names):
            pred_t = model_var[:, idx]
            target_t = targets[:, idx]
            task_losses[name] = tf.reduce_mean(tf.square(pred_t - target_t))

        total_loss = layer(task_losses)

    # Gradients with respect to model variable
    grad_model = tape.gradient(total_loss, model_var)
    assert grad_model is not None
    assert grad_model.shape == (4, 7)
    assert not np.isnan(grad_model.numpy()).any()
    # Predictions are 1.0, targets are 0.0, d(loss)/d(pred) = 2.0 * 1.0 / 4 * 4 = 2.0
    # exp(-0) * 2.0 = 2.0 (scaled by batch dimension)
    assert np.all(grad_model.numpy() > 0.0)

    # Gradients with respect to layer log_vars (s_t)
    grad_log_vars = tape.gradient(total_loss, layer.trainable_weights[0])
    assert grad_log_vars is not None
    assert grad_log_vars.shape == (7,)
    assert not np.isnan(grad_log_vars.numpy()).any()

    # Analytical check: d(total_loss)/ds_t = -exp(-s_t) * L_t + 0.5
    # Here L_t = 1.0, s_t = 0.0 => -1.0 + 0.5 = -0.5 for all tasks
    expected_grad_s = np.full((7,), -0.5, dtype=np.float32)
    np.testing.assert_allclose(grad_log_vars.numpy(), expected_grad_s, atol=1e-5)


def test_homoscedastic_loss_serialization():
    layer = HomoscedasticLossLayer(task_names=list(PRUNED_LABEL_KEYS), name="custom_homoscedastic")
    # Build the layer
    dummy_losses = {k: tf.constant(1.5, dtype=tf.float32) for k in PRUNED_LABEL_KEYS}
    _ = layer(dummy_losses)

    config = layer.get_config()
    assert "task_names" in config
    assert config["task_names"] == list(PRUNED_LABEL_KEYS)
    assert config["name"] == "custom_homoscedastic"

    # Deserialization
    restored_layer = HomoscedasticLossLayer.from_config(config)
    restored_loss = restored_layer(dummy_losses)
    np.testing.assert_allclose(restored_loss.numpy(), layer(dummy_losses).numpy(), atol=1e-5)
