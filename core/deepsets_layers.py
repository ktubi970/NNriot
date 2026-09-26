"""Permutation-invariant and Siamese DeepSets layers for 5v5 team drafts.

Based on Zaheer et al. (2017) "Deep Sets" and Lee et al. (2019) "Set Transformer".
Guarantees exact permutation invariance within teams and exact anti-symmetry
under team swapping: T_diff(B, A) = -T_diff(A, B).
"""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="nnriot")
class SiameseTeamEncoder(tf.keras.layers.Layer):
    """Encodes two 5-player teams into permutation-invariant, anti-symmetric representations.
    
    Inputs:
        tuple (team_a, team_b) each of shape (batch, 5, feature_dim).
        
    Outputs:
        (T_diff, T_sum):
            T_diff has shape (batch, latent_dim), strictly anti-symmetric:
                T_diff(B, A) == -T_diff(A, B)
            T_sum has shape (batch, latent_dim), strictly symmetric:
                T_sum(B, A) == T_sum(A, B)
    """

    def __init__(
        self,
        embed_dim: int = 32,
        latent_dim: int = 64,
        dropout_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.dropout_rate = float(dropout_rate)

        # phi: per-entity feature extractor (shared across all 10 players)
        self.phi1 = tf.keras.layers.Dense(self.embed_dim, activation="relu", name="phi_dense1")
        self.phi2 = tf.keras.layers.Dense(self.embed_dim, activation="relu", name="phi_dense2")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

        # rho: team-level representation transformer (shared for Team A and Team B)
        self.rho1 = tf.keras.layers.Dense(self.latent_dim, activation="relu", name="rho_dense1")
        self.rho2 = tf.keras.layers.Dense(self.latent_dim, activation=None, name="rho_dense2")

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor], training: bool | None = None) -> tuple[tf.Tensor, tf.Tensor]:
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError(f"SiameseTeamEncoder expects a 2-tuple of (team_a, team_b), got {inputs}")

        team_a, team_b = inputs[0], inputs[1]

        # 1. Entity embedding phi(x) for each player in Team A and Team B
        # Input shape: (batch, 5, feature_dim) -> (batch, 5, embed_dim)
        h_a = self.phi2(self.phi1(team_a))
        h_b = self.phi2(self.phi1(team_b))

        if self.dropout is not None:
            h_a = self.dropout(h_a, training=training)
            h_b = self.dropout(h_b, training=training)

        # 2. DeepSets permutation-invariant sum-pooling across the 5 players
        # Shape: (batch, embed_dim)
        pool_a = tf.reduce_sum(h_a, axis=1)
        pool_b = tf.reduce_sum(h_b, axis=1)

        # 3. Team-level representation rho(sum)
        # Shape: (batch, latent_dim)
        t_a = self.rho2(self.rho1(pool_a))
        t_b = self.rho2(self.rho1(pool_b))

        # 4. Anti-symmetric difference and symmetric sum
        t_diff = tf.subtract(t_b, t_a, name="t_diff")
        t_sum = tf.add(t_a, t_b, name="t_sum")

        return t_diff, t_sum

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config
