import pytest
import numpy as np
import tensorflow as tf
from core.cagrad import project_conflict_averse_gradients
from core.pcgrad import cagrad_step

def test_cagrad_resolves_conflict():
    # Deux tâches avec gradients opposés
    # g1 = [1.0, 0.0], g2 = [-1.0, 0.5]
    v = tf.Variable([0.0, 0.0])
    g1 = [tf.constant([1.0, 0.0], dtype=tf.float32)]
    g2 = [tf.constant([-1.0, 0.5], dtype=tf.float32)]
    
    combined = project_conflict_averse_gradients([g1, g2], [v], c=0.5)
    assert len(combined) == 1
    g_res = combined[0].numpy()
    
    # Le gradient combiné ne doit pas avoir un produit scalaire négatif avec la pire tâche au-delà de la borne
    dot1 = np.dot(g_res, [1.0, 0.0])
    dot2 = np.dot(g_res, [-1.0, 0.5])
    assert dot1 >= -0.1
    assert dot2 >= -0.1
    assert not np.isnan(g_res).any()

def test_cagrad_non_conflicting():
    # Tâches avec gradients dans la même direction
    v = tf.Variable([0.0, 0.0])
    g1 = [tf.constant([1.0, 1.0], dtype=tf.float32)]
    g2 = [tf.constant([0.5, 0.5], dtype=tf.float32)]
    
    combined = project_conflict_averse_gradients([g1, g2], [v], c=0.5)
    assert len(combined) == 1
    g_res = combined[0].numpy()
    assert not np.isnan(g_res).any()
    # Gradients should point in positive quadrant
    assert g_res[0] > 0.0 and g_res[1] > 0.0

def test_cagrad_edge_cases():
    v = tf.Variable([0.0, 0.0])
    # Empty gradients
    assert project_conflict_averse_gradients([], [v]) == []
    
    # Single task
    g1 = [tf.constant([1.0, 2.0], dtype=tf.float32)]
    res_single = project_conflict_averse_gradients([g1], [v])
    np.testing.assert_allclose(res_single[0].numpy(), [1.0, 2.0], atol=1e-5)
    
    # Zero gradients
    g_zero1 = [tf.constant([0.0, 0.0], dtype=tf.float32)]
    g_zero2 = [tf.constant([0.0, 0.0], dtype=tf.float32)]
    res_zero = project_conflict_averse_gradients([g_zero1, g_zero2], [v])
    np.testing.assert_allclose(res_zero[0].numpy(), [0.0, 0.0], atol=1e-5)

def test_cagrad_multi_variables_and_none_vars():
    v1 = tf.Variable(tf.zeros((2, 2)))
    v2 = tf.Variable(tf.zeros((2,)))
    
    g1_v1 = tf.constant([[1.0, -1.0], [0.5, -0.5]], dtype=tf.float32)
    g1_v2 = tf.constant([0.2, -0.2], dtype=tf.float32)
    
    g2_v1 = tf.constant([[-1.0, 1.0], [-0.5, 0.5]], dtype=tf.float32)
    g2_v2 = tf.constant([-0.2, 0.2], dtype=tf.float32)
    
    # Test with variables provided
    res_with_vars = project_conflict_averse_gradients([[g1_v1, g1_v2], [g2_v1, g2_v2]], [v1, v2], c=0.5)
    assert len(res_with_vars) == 2
    assert res_with_vars[0].shape == (2, 2)
    assert res_with_vars[1].shape == (2,)
    
    # Test without variables provided (inferred)
    res_inferred = project_conflict_averse_gradients([[g1_v1, g1_v2], [g2_v1, g2_v2]], c=0.5)
    assert len(res_inferred) == 2
    assert res_inferred[0].shape == (2, 2)
    assert res_inferred[1].shape == (2,)

def test_cagrad_scale_invariance():
    # Scale invariance test: ||g|| = 10^-5 vs ||g|| = 1.0 vs ||g|| = 10^3
    v = tf.Variable([0.0, 0.0])
    g1 = [tf.constant([1.0, 0.0], dtype=tf.float32)]
    g2 = [tf.constant([-1.0, 0.5], dtype=tf.float32)]
    
    res_base = project_conflict_averse_gradients([g1, g2], [v], c=0.5)[0].numpy()
    
    # Scale very small: 1e-5 (near convergence)
    g1_small = [tf.constant([1e-5, 0.0], dtype=tf.float32)]
    g2_small = [tf.constant([-1e-5, 0.5e-5], dtype=tf.float32)]
    res_small = project_conflict_averse_gradients([g1_small, g2_small], [v], c=0.5)[0].numpy()
    np.testing.assert_allclose(res_small / 1e-5, res_base, rtol=1e-4, atol=1e-5)
    
    # Scale very large: 1e3
    g1_large = [tf.constant([1e3, 0.0], dtype=tf.float32)]
    g2_large = [tf.constant([-1e3, 0.5e3], dtype=tf.float32)]
    res_large = project_conflict_averse_gradients([g1_large, g2_large], [v], c=0.5)[0].numpy()
    np.testing.assert_allclose(res_large / 1e3, res_base, rtol=1e-4, atol=1e-5)

def test_cagrad_task_permutation_invariance():
    # Permuting tasks [task1, task2] vs [task2, task1] must produce identical combined gradient
    v = tf.Variable([0.0, 0.0])
    g1 = [tf.constant([1.0, 0.0], dtype=tf.float32)]
    g2 = [tf.constant([-1.0, 0.5], dtype=tf.float32)]
    
    res_12 = project_conflict_averse_gradients([g1, g2], [v], c=0.5)[0].numpy()
    res_21 = project_conflict_averse_gradients([g2, g1], [v], c=0.5)[0].numpy()
    
    np.testing.assert_allclose(res_12, res_21, rtol=1e-5, atol=1e-5)

def test_cagrad_step_execution():
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
    
    losses = cagrad_step(model, optimizer, x, y, loss_fns, c=0.5)
    assert "head1" in losses
    assert "head2" in losses
    assert losses["head1"] >= 0.0
    assert losses["head2"] >= 0.0
