"""Permutation-invariant and Siamese DeepSets layers for 5v5 team drafts.

Based on:
- Zaheer et al. (2017) "Deep Sets" (NeurIPS 2017)
- Lee et al. (2019) "Set Transformer" (ICML 2019)
- Rendle (2010) "Factorization Machines" (Bilinear synergy)

Guarantees exact mathematical permutation invariance within teams and exact
anti-symmetry under team swapping: T_diff(B, A) = -T_diff(A, B).
"""
from __future__ import annotations

from typing import Any
import tensorflow as tf

from core.champion_registry import NUM_CHAMPIONS


@tf.keras.utils.register_keras_serializable(package="nnriot")
class SiameseTeamEncoder(tf.keras.layers.Layer):
    """Encodes two 5-player teams into permutation-invariant, anti-symmetric representations.
    
    Inputs:
        tuple (team_a, team_b) each of shape (batch, 5, 8).
        Layout per player: [champion_id, role_id, smooth_wr, smooth_cwr, champ_exp, rec_wr, role_fam, tot_exp]
        
    Outputs:
        (T_diff, T_sum, T_a, T_b):
            T_diff has shape (batch, total_diff_dim), strictly anti-symmetric:
                T_diff(B, A) == -T_diff(A, B)
            T_sum has shape (batch, total_sum_dim), strictly symmetric:
                T_sum(B, A) == T_sum(A, B)
            T_a and T_b have shape (batch, team_dim), individual team representations.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        latent_dim: int = 64,
        champ_embed_dim: int = 32,
        role_embed_dim: int = 8,
        lane_proj_dim: int = 16,
        dropout_rate: float = 0.0,
        l2_reg: float = 1e-4,
        num_champions: int = NUM_CHAMPIONS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.champ_embed_dim = int(champ_embed_dim)
        self.role_embed_dim = int(role_embed_dim)
        self.lane_proj_dim = int(lane_proj_dim)
        self.dropout_rate = float(dropout_rate)
        self.l2_reg = float(l2_reg)
        self.num_champions = int(num_champions)

        reg = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None

        # 1. Learnable embeddings for discrete entities with L2 regularizer
        self.champ_embed = tf.keras.layers.Embedding(
            self.num_champions, self.champ_embed_dim,
            embeddings_regularizer=reg,
            name="champ_embedding"
        )
        self.role_embed = tf.keras.layers.Embedding(
            5, self.role_embed_dim,
            embeddings_regularizer=reg,
            name="role_embedding"
        )

        # 2. phi: per-player entity feature extractor (shared across all 10 players)
        self.phi1 = tf.keras.layers.Dense(self.embed_dim, activation="relu", kernel_regularizer=reg, name="phi_dense1")
        self.phi2 = tf.keras.layers.Dense(self.embed_dim, activation="relu", kernel_regularizer=reg, name="phi_dense2")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

        # 3. rho: DeepSets team representation transformer (shared for Team A and Team B)
        self.rho1 = tf.keras.layers.Dense(self.latent_dim, activation="relu", kernel_regularizer=reg, name="rho_dense1")
        self.rho2 = tf.keras.layers.Dense(self.latent_dim, activation=None, kernel_regularizer=reg, name="rho_dense2")

        # 4. Bilinear champion synergy projection (Factorization Machines)
        self.synergy_proj = tf.keras.layers.Dense(self.latent_dim, activation=None, kernel_regularizer=reg, name="synergy_proj")

        # 5. Lane-to-lane direct matchup projections (NO BIAS to preserve exact anti-symmetry)
        self.lane_diff_proj = tf.keras.layers.Dense(
            self.lane_proj_dim, use_bias=False, activation=None, kernel_regularizer=reg, name="lane_diff_proj"
        )
        self.lane_sum_proj = tf.keras.layers.Dense(
            self.lane_proj_dim, use_bias=True, activation="relu", kernel_regularizer=reg, name="lane_sum_proj"
        )

    def build(self, input_shape: Any) -> None:
        self.champ_embed.build((None, 5))
        self.role_embed.build((None, 5))
        in_dim = self.champ_embed_dim + self.role_embed_dim + 6
        self.phi1.build((None, 5, in_dim))
        self.phi2.build((None, 5, self.embed_dim))
        self.rho1.build((None, self.embed_dim))
        self.rho2.build((None, self.latent_dim))
        self.synergy_proj.build((None, self.champ_embed_dim))
        self.lane_diff_proj.build((None, 5, self.embed_dim))
        self.lane_sum_proj.build((None, 5, self.embed_dim))
        super().build(input_shape)

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
        training: bool | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError(f"SiameseTeamEncoder expects a 2-tuple of (team_a, team_b), got {inputs}")

        team_a, team_b = inputs[0], inputs[1]

        def _process_team(team: tf.Tensor):
            # Sort players into canonical role order [0..4] (TOP, JNG, MID, BOT, SUP)
            # so lane matchups are role-aligned and within-team permutations are strictly invariant
            role_indices = tf.clip_by_value(tf.cast(tf.round(team[:, :, 1]), tf.int32), 0, 4)
            order = tf.argsort(role_indices, axis=1)
            ordered_team = tf.gather(team, order, batch_dims=1)

            c_ids = tf.clip_by_value(tf.cast(tf.round(ordered_team[:, :, 0]), tf.int32), 0, self.num_champions - 1)
            r_ids = tf.clip_by_value(tf.cast(tf.round(ordered_team[:, :, 1]), tf.int32), 0, 4)
            priors = ordered_team[:, :, 2:]

            e_champ = self.champ_embed(c_ids)  # (batch, 5, champ_embed_dim)
            e_role = self.role_embed(r_ids)    # (batch, 5, role_embed_dim)

            entity_in = tf.concat([e_champ, e_role, priors], axis=-1)
            h = self.phi2(self.phi1(entity_in))  # (batch, 5, embed_dim)

            if self.dropout is not None:
                h = self.dropout(h, training=training)

            # DeepSets permutation-invariant sum pool
            pool = tf.reduce_sum(h, axis=1)     # (batch, embed_dim)
            t = self.rho2(self.rho1(pool))      # (batch, latent_dim)

            # Bilinear champion synergy (FM pooling: 0.5 * ((sum v)^2 - sum v^2))
            sum_c = tf.reduce_sum(e_champ, axis=1)
            sum_sq_c = tf.reduce_sum(tf.square(e_champ), axis=1)
            synergy = 0.5 * (tf.square(sum_c) - sum_sq_c)
            syn_feat = self.synergy_proj(synergy)  # (batch, latent_dim)

            return h, t, syn_feat

        h_a, t_a_pool, syn_a = _process_team(team_a)
        h_b, t_b_pool, syn_b = _process_team(team_b)

        # 1. Lane matchup differences (Team B - Team A for each lane 0..4)
        # Using a shared linear projection with NO BIAS to preserve strict anti-symmetry
        lane_diff = h_b - h_a  # (batch, 5, embed_dim)
        lane_diff_proj = self.lane_diff_proj(lane_diff)  # (batch, 5, lane_proj_dim)
        lane_diff_flat = tf.reshape(lane_diff_proj, [-1, 5 * self.lane_proj_dim])

        lane_sum = h_a + h_b  # (batch, 5, embed_dim)
        lane_sum_proj = self.lane_sum_proj(lane_sum)
        lane_sum_flat = tf.reshape(lane_sum_proj, [-1, 5 * self.lane_proj_dim])

        # 2. Team DeepSet differences and synergy differences
        pool_diff = t_b_pool - t_a_pool
        syn_diff = syn_b - syn_a

        pool_sum = t_a_pool + t_b_pool
        syn_sum = syn_a + syn_b

        # 3. Strictly anti-symmetric combined difference: T_diff(B, A) == -T_diff(A, B)
        t_diff = tf.concat([pool_diff, syn_diff, lane_diff_flat], axis=-1, name="t_diff")

        # 4. Strictly symmetric combined sum: T_sum(B, A) == T_sum(A, B)
        t_sum = tf.concat([pool_sum, syn_sum, lane_sum_flat], axis=-1, name="t_sum")

        # 5. Team representations
        t_a = tf.concat([t_a_pool, syn_a], axis=-1, name="t_a")
        t_b = tf.concat([t_b_pool, syn_b], axis=-1, name="t_b")

        return t_diff, t_sum, t_a, t_b

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "champ_embed_dim": self.champ_embed_dim,
            "role_embed_dim": self.role_embed_dim,
            "lane_proj_dim": self.lane_proj_dim,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "num_champions": self.num_champions,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="nnriot")
class AntiSymmetricSoftmax(tf.keras.layers.Layer):
    """Takes a scalar logit s and outputs 2-class softmax [-s, s], strictly anti-symmetric."""

    def call(self, s: tf.Tensor) -> tf.Tensor:
        neg_s = -s
        logits = tf.concat([neg_s, s], axis=-1)
        return tf.nn.softmax(logits, axis=-1)
