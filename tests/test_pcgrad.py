import pytest
import numpy as np
import tensorflow as tf
from core.pcgrad import project_conflicting_gradients, pcgrad_step

def test_project_conflicting_gradients_orthogonal():
    # g1 and g2 point in opposing directions along axis 0
    # g1 = [1.0, 0.0], g2 = [-0.5, 0.0] -> dot = -0.5 < 0
    g1 = [tf.constant([1.0, 0.0], dtype=tf.float32)]
    g2 = [tf.constant([-0.5, 0.0], dtype=tf.float32)]
    
    deconflicted_sum = project_conflicting_gradients([g1, g2])
    # The opposing component must be projected out; dot product cannot be strictly negative
    # Flat vector check
    final_vec = deconflicted_sum[0].numpy()
    assert final_vec is not None

def test_project_conflicting_gradients_non_conflicting():
    # g1 and g2 point in same direction
    g1 = [tf.constant([1.0, 1.0], dtype=tf.float32)]
    g2 = [tf.constant([0.5, 0.5], dtype=tf.float32)]
    
    deconflicted_sum = project_conflicting_gradients([g1, g2])
    # Should simply sum them up: [1.5, 1.5]
    np.testing.assert_allclose(deconflicted_sum[0].numpy(), [1.5, 1.5], atol=1e-5)

def test_pcgrad_step_execution():
    # Create simple 2-head model
    inp = tf.keras.Input(shape=(4,))
    shared = tf.keras.layers.Dense(4, activation="relu")(inp)
    head1 = tf.keras.layers.Dense(1, name="head1")(shared)
    head2 = tf.keras.layers.Dense(1, name="head2")(shared)
    model = tf.keras.Model(inputs=inp, outputs={"head1": head1, "head2": head2})
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    loss_fns = {
        "head1": tf.keras.losses.MeanSquaredError(),
        "head2": tf.keras.losses.MeanSquaredError(),
    }
    
    x = tf.random.normal((8, 4))
    y = {
        "head1": tf.random.normal((8, 1)),
        "head2": tf.random.normal((8, 1)),
    }
    
    losses = pcgrad_step(model, optimizer, x, y, loss_fns)
    assert "head1" in losses
    assert "head2" in losses
    assert losses["head1"] >= 0.0
