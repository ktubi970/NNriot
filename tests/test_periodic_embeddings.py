"""Tests for PeriodicNumericalEmbedding layer (Fourier feature embeddings for continuous tabular features).
"""
import math
import tempfile
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from core.periodic_embeddings import PeriodicNumericalEmbedding


def test_periodic_embedding_shape_3d():
    """Verify output shape on 3D team features: (batch, 5, num_features)."""
    layer = PeriodicNumericalEmbedding(num_features=12, num_frequencies=8)
    inputs = tf.random.normal((4, 5, 12))
    out = layer(inputs)

    # 12 features * 2 (sin/cos) * 8 frequencies = 192
    assert out.shape == (4, 5, 12 * 16)
    assert out.shape == (4, 5, 192)


def test_periodic_embedding_shape_2d():
    """Verify output shape on 2D flat features: (batch, num_features)."""
    layer = PeriodicNumericalEmbedding(num_features=12, num_frequencies=8)
    inputs = tf.random.normal((4, 12))
    out = layer(inputs)

    assert out.shape == (4, 12 * 16)
    assert out.shape == (4, 192)


def test_periodic_embedding_gradients():
    """Verify non-zero gradients backpropagate through learnable coefficients."""
    layer = PeriodicNumericalEmbedding(num_features=12, num_frequencies=8)
    inputs = tf.random.normal((4, 5, 12))

    with tf.GradientTape() as tape:
        out = layer(inputs)
        loss = tf.reduce_mean(out)

    grads = tape.gradient(loss, layer.trainable_weights)
    assert len(grads) == len(layer.trainable_weights) == 1
    assert grads[0] is not None
    assert grads[0].shape == (12, 8)
    # Gradient should not be all zeros
    assert float(tf.reduce_sum(tf.abs(grads[0]))) > 0.0


def test_periodic_embedding_exact_math():
    """Verify exact mathematical computation: [sin(2*pi*c*x), cos(2*pi*c*x)]."""
    layer = PeriodicNumericalEmbedding(num_features=2, num_frequencies=1)
    # Build explicitly
    layer.build(input_shape=(None, 2))

    # Set known coefficients: feature 0 -> c=1.0, feature 1 -> c=2.0
    layer.coefficients.assign(np.array([[1.0], [2.0]], dtype=np.float32))

    # Inputs: x0 = 0.25 -> 2*pi*1.0*0.25 = pi/2 -> sin=1, cos=0
    #         x1 = 0.125 -> 2*pi*2.0*0.125 = pi/2 -> sin=1, cos=0
    inputs = tf.constant([[0.25, 0.125]], dtype=tf.float32)
    out = layer(inputs)

    expected = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)

    # Test with x0=0 -> sin(0)=0, cos(0)=1
    #           x1=0.25 -> 2*pi*2.0*0.25 = pi -> sin(pi)=0, cos(pi)=-1
    inputs_zero = tf.constant([[0.0, 0.25]], dtype=tf.float32)
    out_zero = layer(inputs_zero)
    expected_zero = np.array([[0.0, 1.0, 0.0, -1.0]], dtype=np.float32)
    np.testing.assert_allclose(out_zero.numpy(), expected_zero, atol=1e-5)


def test_periodic_embedding_serialization_layer():
    """Verify layer serialization via serialize and deserialize."""
    layer = PeriodicNumericalEmbedding(
        num_features=10,
        num_frequencies=4,
        sigma=0.5,
    )
    layer.build(input_shape=(None, 5, 10))

    serialized = tf.keras.layers.serialize(layer)
    reconstructed = tf.keras.layers.deserialize(serialized)

    assert isinstance(reconstructed, PeriodicNumericalEmbedding)
    assert reconstructed.num_features == 10
    assert reconstructed.num_frequencies == 4
    assert reconstructed.sigma == 0.5


def test_periodic_embedding_model_save_load(tmp_path: Path):
    """Verify end-to-end model saving and loading with PeriodicNumericalEmbedding."""
    inputs = tf.keras.layers.Input(shape=(5, 12), name="cont_inputs")
    embedded = PeriodicNumericalEmbedding(num_features=12, num_frequencies=4)(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(embedded)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    x_test = tf.random.normal((2, 5, 12))
    pred_orig = model(x_test).numpy()

    save_path = str(tmp_path / "test_periodic_model.keras")
    model.save(save_path)

    loaded_model = tf.keras.models.load_model(save_path)
    pred_loaded = loaded_model(x_test).numpy()

    np.testing.assert_allclose(pred_orig, pred_loaded, atol=1e-6)


def test_periodic_embedding_trainable_flag():
    """Verify trainable=False disables gradient tracking on coefficients."""
    layer = PeriodicNumericalEmbedding(num_features=6, num_frequencies=4, trainable=False)
    layer.build(input_shape=(None, 6))

    assert len(layer.trainable_weights) == 0
    assert len(layer.non_trainable_weights) == 1
    assert layer.coefficients.trainable is False


def test_periodic_embedding_invalid_shape():
    """Verify mismatched last dimension raises ValueError."""
    layer = PeriodicNumericalEmbedding(num_features=12, num_frequencies=8)
    bad_inputs = tf.random.normal((2, 5, 10))

    with pytest.raises(ValueError, match="num_features"):
        layer(bad_inputs)
