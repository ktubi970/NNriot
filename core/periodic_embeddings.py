"""Periodic Numerical Embeddings for continuous features in Deep Learning.

Based on Gorishniy et al. (NeurIPS 2022) "On Embeddings for Numerical Features in Tabular Data".
Transforms continuous features into Fourier sinusoidal representations:
    v(x) = [sin(2*pi*c*x), cos(2*pi*c*x)]
where c are learnable or fixed frequency multipliers.
"""
from __future__ import annotations

import math
from typing import Any
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="nnriot")
class PeriodicNumericalEmbedding(tf.keras.layers.Layer):
    """Encodes scalar numerical features into sinusoidal Fourier embeddings.

    For each input scalar feature x_i, computes:
        [sin(2*pi*c_{i,1}*x_i), ..., sin(2*pi*c_{i,k}*x_i),
         cos(2*pi*c_{i,1}*x_i), ..., cos(2*pi*c_{i,k}*x_i)]
    where k = num_frequencies.

    Input shape:
        - 2D tensor: `(batch, num_features)`
        - 3D tensor: `(batch, 5, num_features)` (or arbitrary `(*batch_dims, num_features)`)

    Output shape:
        - 2D tensor: `(batch, num_features * 2 * num_frequencies)`
        - 3D tensor: `(batch, 5, num_features * 2 * num_frequencies)`
    """

    def __init__(
        self,
        num_features: int = 12,
        num_frequencies: int = 8,
        sigma: float = 1.0,
        trainable: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(trainable=trainable, **kwargs)
        if int(num_features) <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if int(num_frequencies) <= 0:
            raise ValueError(f"num_frequencies must be positive, got {num_frequencies}")
        if float(sigma) <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.num_features = int(num_features)
        self.num_frequencies = int(num_frequencies)
        self.sigma = float(sigma)

    def build(self, input_shape: Any = None) -> None:
        # Frequency coefficients c_i,j for each feature i and frequency j
        self.coefficients = self.add_weight(
            name="coefficients",
            shape=(self.num_features, self.num_frequencies),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.sigma),
            trainable=self.trainable,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.convert_to_tensor(inputs, dtype=self.compute_dtype or tf.float32)
        if inputs.shape[-1] is not None and inputs.shape[-1] != self.num_features:
            raise ValueError(
                f"Input last dimension ({inputs.shape[-1]}) does not match "
                f"expected num_features ({self.num_features})."
            )

        # Expand last dimension for broadcasting across frequencies:
        # shape: (..., num_features, 1) * (num_features, num_frequencies) -> (..., num_features, num_frequencies)
        x_expanded = tf.expand_dims(inputs, axis=-1)
        angles = 2.0 * math.pi * x_expanded * self.coefficients
        sin_part = tf.sin(angles)
        cos_part = tf.cos(angles)

        # Concatenate sin and cos along frequency axis:
        # shape: (..., num_features, 2 * num_frequencies)
        fourier = tf.concat([sin_part, cos_part], axis=-1)

        # Flatten feature and frequency dimensions:
        lead_shape = tf.shape(inputs)[:-1]
        out_dim = self.num_features * (2 * self.num_frequencies)
        target_shape = tf.concat([lead_shape, [out_dim]], axis=0)

        out = tf.reshape(fourier, target_shape)
        if inputs.shape.rank is not None:
            out.set_shape(self.compute_output_shape(inputs.shape))
        return out

    def compute_output_shape(self, input_shape: Any) -> tf.TensorShape:
        input_shape = tf.TensorShape(input_shape)
        out_dim = self.num_features * (2 * self.num_frequencies)
        return input_shape[:-1].concatenate([out_dim])

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "num_frequencies": self.num_frequencies,
            "sigma": self.sigma,
        })
        return config
