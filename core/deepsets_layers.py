"""Permutation-invariant and Siamese DeepSets layers for 5v5 team drafts.

Based on:
- Zaheer et al. (2017) "Deep Sets" (NeurIPS 2017)
- Lee et al. (2019) "Set Transformer" (ICML 2019)
- Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NeurIPS 2013)
- Rendle (2010) "Factorization Machines" (Bilinear synergy)

Guarantees exact mathematical permutation invariance within teams and exact
anti-symmetry under team swapping: T_diff(B, A) = -T_diff(A, B).
"""
from __future__ import annotations

from typing import Any
import tensorflow as tf

from core.champion_registry import NUM_CHAMPIONS
from core.periodic_embeddings import PeriodicNumericalEmbedding


def _get_last_dim(input_shape: Any, default_dim: int) -> int:
    """Safely extract the last feature dimension from tensor or tuple shapes."""
    if input_shape is None:
        return default_dim
    if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list, tf.TensorShape)):
        shape = input_shape[0]
    else:
        shape = input_shape
    if hasattr(shape, "__getitem__") and len(shape) > 0 and shape[-1] is not None:
        return int(shape[-1])
    return default_dim


@tf.keras.utils.register_keras_serializable(package="nnriot")
class SetTransformerPMA(tf.keras.layers.Layer):
    """Pooling by Multihead Attention (PMA) with Set Attention Block (SAB).

    Lee et al. (ICML 2019) "Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks".

    Computes intra-team self-attention across 5 player tokens followed by
    multi-head attention pooling with k learned seed query vectors. Guarantees
    strict permutation invariance over the input set.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 2,
        num_seeds: int = 2,
        use_sab: bool = True,
        dropout_rate: float = 0.0,
        l2_reg: float = 1e-4,
        flatten: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_seeds = int(num_seeds)
        self.use_sab = bool(use_sab)
        self.dropout_rate = float(dropout_rate)
        self.l2_reg = float(l2_reg)
        self.flatten = bool(flatten)

        reg = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        key_dim = max(4, self.embed_dim // self.num_heads)

        if self.use_sab:
            self.sab_mha = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=key_dim, name="sab_mha"
            )
            self.sab_norm1 = tf.keras.layers.LayerNormalization(name="sab_norm1")
            self.sab_ffn = tf.keras.layers.Dense(
                self.embed_dim, activation="relu", kernel_regularizer=reg, name="sab_ffn"
            )
            self.sab_norm2 = tf.keras.layers.LayerNormalization(name="sab_norm2")

        self.pma_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=key_dim, name="pma_mha"
        )
        self.pma_norm1 = tf.keras.layers.LayerNormalization(name="pma_norm1")
        self.pma_ffn = tf.keras.layers.Dense(
            self.embed_dim, activation="relu", kernel_regularizer=reg, name="pma_ffn"
        )
        self.pma_norm2 = tf.keras.layers.LayerNormalization(name="pma_norm2")

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

    def build(self, input_shape: Any = None) -> None:
        if self.built:
            return
        in_dim = _get_last_dim(input_shape, self.embed_dim)
        self.seed_vectors = self.add_weight(
            name="seed_vectors",
            shape=(1, self.num_seeds, self.embed_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=self.trainable,
        )
        if self.use_sab:
            self.sab_mha.build((None, 5, in_dim), (None, 5, in_dim))
            self.sab_norm1.build((None, 5, in_dim))
            self.sab_ffn.build((None, 5, in_dim))
            self.sab_norm2.build((None, 5, self.embed_dim))

        self.pma_mha.build((None, self.num_seeds, self.embed_dim), (None, 5, self.embed_dim))
        self.pma_norm1.build((None, self.num_seeds, self.embed_dim))
        self.pma_ffn.build((None, self.num_seeds, self.embed_dim))
        self.pma_norm2.build((None, self.num_seeds, self.embed_dim))
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        # inputs shape: (batch, 5, embed_dim)
        if self.use_sab:
            attn_out = self.sab_mha(inputs, inputs, training=training)
            if self.dropout is not None:
                attn_out = self.dropout(attn_out, training=training)
            h = self.sab_norm1(inputs + attn_out)
            ffn_out = self.sab_ffn(h)
            if self.dropout is not None:
                ffn_out = self.dropout(ffn_out, training=training)
            h = self.sab_norm2(h + ffn_out)
        else:
            h = inputs

        batch_size = tf.shape(inputs)[0]
        seeds = tf.tile(self.seed_vectors, [batch_size, 1, 1])

        pma_out = self.pma_mha(query=seeds, key=h, value=h, training=training)
        if self.dropout is not None:
            pma_out = self.dropout(pma_out, training=training)
        pma_h = self.pma_norm1(seeds + pma_out)
        pma_ffn_out = self.pma_ffn(pma_h)
        if self.dropout is not None:
            pma_ffn_out = self.dropout(pma_ffn_out, training=training)
        pooled = self.pma_norm2(pma_h + pma_ffn_out)

        if self.flatten:
            return tf.reshape(pooled, [-1, self.num_seeds * self.embed_dim])
        return pooled

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_seeds": self.num_seeds,
            "use_sab": self.use_sab,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "flatten": self.flatten,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="nnriot")
class SinkhornCrossAttention(tf.keras.layers.Layer):
    """Doubly stochastic optimal transport cross-attention via Sinkhorn-Knopp.

    Guarantees:
    1. Doubly stochastic transport plan: row sums = 1.0, col sums = 1.0.
    2. Strict anti-symmetry under team swap:
       P(team_b, team_a) == P(team_a, team_b)^T
       attn_a(team_b, team_a) == attn_b(team_a, team_b)
       attn_b(team_b, team_a) == attn_a(team_a, team_b)
    """

    def __init__(
        self,
        embed_dim: int = 32,
        temperature: float = 1.0,
        num_iterations: int = 10,
        l2_reg: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.temperature = float(temperature)
        self.num_iterations = int(num_iterations)
        self.l2_reg = float(l2_reg)

        reg = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        # Shared projection without bias to ensure exact symmetry under swap
        self.proj_qk = tf.keras.layers.Dense(
            self.embed_dim, use_bias=False, kernel_regularizer=reg, name="proj_qk"
        )
        self.proj_v = tf.keras.layers.Dense(
            self.embed_dim, use_bias=False, kernel_regularizer=reg, name="proj_v"
        )

    def build(self, input_shape: Any = None) -> None:
        if self.built:
            return
        in_dim = _get_last_dim(input_shape, self.embed_dim)
        if not self.proj_qk.built:
            self.proj_qk.build((None, 5, in_dim))
        if not self.proj_v.built:
            self.proj_v.build((None, 5, in_dim))
        super().build(input_shape)

    def _sinkhorn_iterations(self, log_cost: tf.Tensor) -> tf.Tensor:
        """Run Sinkhorn-Knopp iterations in log-space for numerical stability."""
        log_p = log_cost
        for _ in range(self.num_iterations):
            log_p = log_p - tf.reduce_logsumexp(log_p, axis=-1, keepdims=True)
            log_p = log_p - tf.reduce_logsumexp(log_p, axis=-2, keepdims=True)
        return tf.exp(log_p)

    def compute_transport_plan(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        """Compute the doubly stochastic transport plan matrix of shape (batch, 5, 5)."""
        if not self.built:
            in_dim = int(x1.shape[-1]) if x1.shape[-1] is not None else self.embed_dim
            self.build([(None, 5, in_dim), (None, 5, in_dim)])

        q1 = tf.math.l2_normalize(self.proj_qk(x1), axis=-1)
        k2 = tf.math.l2_normalize(self.proj_qk(x2), axis=-1)
        cost = tf.matmul(q1, k2, transpose_b=True) / self.temperature

        # Symmetrized Sinkhorn to guarantee exact transpose equivariance under team swap:
        # P(B, A) == P(A, B)^T to machine precision
        p1 = self._sinkhorn_iterations(cost)
        cost_t = tf.transpose(cost, [0, 2, 1])
        p2 = tf.transpose(self._sinkhorn_iterations(cost_t), [0, 2, 1])
        return 0.5 * (p1 + p2)

    def call(
        self,
        x1: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
        x2: tf.Tensor | None = None,
        return_plan: bool = False,
        training: bool | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if x2 is None:
            if isinstance(x1, (tuple, list)) and len(x1) == 2:
                t1, t2 = x1[0], x1[1]
            else:
                raise ValueError("SinkhornCrossAttention expects (x1, x2) or a 2-element tuple/list.")
        else:
            t1, t2 = x1, x2

        plan = self.compute_transport_plan(t1, t2)
        v1 = self.proj_v(t1)
        v2 = self.proj_v(t2)

        out1 = tf.matmul(plan, v2)
        out2 = tf.matmul(plan, v1, transpose_a=True)

        if return_plan:
            return out1, out2, plan
        return out1, out2

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "temperature": self.temperature,
            "num_iterations": self.num_iterations,
            "l2_reg": self.l2_reg,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="nnriot")
class SiameseTeamEncoder(tf.keras.layers.Layer):
    """Encodes two 5-player teams into permutation-invariant, anti-symmetric representations.

    Architecture:
    - Intra-team SetTransformerPMA (SAB self-attention + PMA with k=2 seeds) replaces/enriches sum pooling.
    - Inter-team SinkhornCrossAttention (doubly stochastic optimal transport) supersedes legacy
      unconstrained MultiHeadAttention (`use_cross_attention`), strictly preserving anti-symmetry
      and mass conservation under team swap.
    - Continuous features are embedded via Fourier PeriodicNumericalEmbedding.

    Inputs:
        tuple (team_a, team_b) each of shape (batch, 5, 8) or (batch, 5, 14).
        Layout per player: [champion_id, role_id, smooth_wr, smooth_cwr, ...]

    Outputs:
        (T_diff, T_sum, T_a, T_b):
            T_diff has shape (batch, total_diff_dim), strictly anti-symmetric:
                T_diff(B, A) == -T_diff(A, B)
            T_sum has shape (batch, total_sum_dim), strictly symmetric:
                T_sum(B, A) == T_sum(A, B)
            T_a and T_b have shape (batch, team_dim), individual team representations.

    Note on Cross Attention:
        `use_sinkhorn_attention` is the modern default (True), enforcing doubly stochastic optimal
        transport with exact transpose-equivariance. The legacy unconstrained `use_cross_attention`
        flag is preserved strictly for backward compatibility and is superseded when `use_sinkhorn_attention`
        is enabled.
    """

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        modern_flags = ('use_set_transformer', 'use_sinkhorn_attention', 'use_periodic_embeddings')
        present = [key in config for key in modern_flags]
        if not any(present):
            # Checkpoints before the SetTransformer/Sinkhorn/Fourier encoder
            # serialized none of these flags. Reconstruct their actual sum-pool
            # architecture rather than adopting today's constructor defaults.
            config.update({key: False for key in modern_flags})
        elif not all(present):
            raise ValueError('Ambiguous SiameseTeamEncoder architecture: incomplete serialized modern flags')
        return cls(**config)

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
        use_cross_attention: bool = False,
        use_set_transformer: bool = True,
        use_sinkhorn_attention: bool = True,
        use_periodic_embeddings: bool = True,
        num_frequencies: int = 8,
        **kwargs: Any,
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
        self.use_cross_attention = bool(use_cross_attention)
        self.use_set_transformer = bool(use_set_transformer)
        self.use_sinkhorn_attention = bool(use_sinkhorn_attention)
        self.use_periodic_embeddings = bool(use_periodic_embeddings)
        self.num_frequencies = int(num_frequencies)

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

        # 2. Periodic embeddings placeholder for continuous features
        self.periodic_embed: PeriodicNumericalEmbedding | None = None

        # 3. phi: per-player entity feature extractor (shared across all 10 players)
        self.phi1 = tf.keras.layers.Dense(self.embed_dim, activation="relu", kernel_regularizer=reg, name="phi_dense1")
        self.phi2 = tf.keras.layers.Dense(self.embed_dim, activation="relu", kernel_regularizer=reg, name="phi_dense2")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

        # 4. SetTransformer PMA (Pooling by Multihead Attention with SAB)
        if self.use_set_transformer:
            self.set_transformer = SetTransformerPMA(
                embed_dim=self.embed_dim,
                num_heads=2,
                num_seeds=2,
                dropout_rate=self.dropout_rate,
                l2_reg=self.l2_reg,
                name="set_transformer_pma",
            )
        else:
            self.set_transformer = None

        # 5. rho: DeepSets team representation transformer (shared for Team A and Team B)
        self.rho1 = tf.keras.layers.Dense(self.latent_dim, activation="relu", kernel_regularizer=reg, name="rho_dense1")
        self.rho2 = tf.keras.layers.Dense(self.latent_dim, activation=None, kernel_regularizer=reg, name="rho_dense2")

        # 6. Bilinear champion synergy projection (Factorization Machines)
        self.synergy_proj = tf.keras.layers.Dense(self.latent_dim, activation=None, kernel_regularizer=reg, name="synergy_proj")

        # 7. Lane-to-lane direct matchup projections (NO BIAS to preserve exact anti-symmetry)
        self.lane_diff_proj = tf.keras.layers.Dense(
            self.lane_proj_dim, use_bias=False, activation=None, kernel_regularizer=reg, name="lane_diff_proj"
        )
        self.lane_sum_proj = tf.keras.layers.Dense(
            self.lane_proj_dim, use_bias=True, activation="relu", kernel_regularizer=reg, name="lane_sum_proj"
        )

        # 8. Cross Attention: Sinkhorn Optimal Transport or unconstrained MHA
        if self.use_sinkhorn_attention:
            self.sinkhorn_attn = SinkhornCrossAttention(
                embed_dim=self.embed_dim,
                temperature=1.0,
                num_iterations=10,
                l2_reg=self.l2_reg,
                name="sinkhorn_cross_attn",
            )
            self.cross_attn = None
        elif self.use_cross_attention:
            self.sinkhorn_attn = None
            self.cross_attn = tf.keras.layers.MultiHeadAttention(
                num_heads=2, key_dim=max(8, self.embed_dim // 2), name="cross_lane_attn"
            )
        else:
            self.sinkhorn_attn = None
            self.cross_attn = None

    def build(self, input_shape: Any = None) -> None:
        if self.built:
            return
        self.champ_embed.build((None, 5))
        self.role_embed.build((None, 5))

        # Dynamic continuous features dimension (total features - champ_id - role_id)
        cont_dim = 12
        if input_shape and isinstance(input_shape, (tuple, list)) and len(input_shape) >= 1:
            first = input_shape[0]
            if hasattr(first, "shape"):
                first = first.shape
            if isinstance(first, (tuple, list)) and len(first) >= 3 and first[-1] is not None:
                cont_dim = int(first[-1]) - 2

        if self.use_periodic_embeddings:
            self.periodic_embed = PeriodicNumericalEmbedding(
                num_features=cont_dim,
                num_frequencies=self.num_frequencies,
                trainable=True,
                name="periodic_embed",
            )
            self.periodic_embed.build((None, 5, cont_dim))
            cont_in_dim = cont_dim * (2 * self.num_frequencies)
        else:
            cont_in_dim = cont_dim

        in_dim = self.champ_embed_dim + self.role_embed_dim + cont_in_dim
        self.phi1.build((None, 5, in_dim))
        self.phi2.build((None, 5, self.embed_dim))

        if self.use_set_transformer and self.set_transformer is not None:
            self.set_transformer.build((None, 5, self.embed_dim))
            pool_in_dim = self.embed_dim + (2 * self.embed_dim)
        else:
            pool_in_dim = self.embed_dim

        self.rho1.build((None, pool_in_dim))
        self.rho2.build((None, self.latent_dim))
        self.synergy_proj.build((None, self.champ_embed_dim))
        self.lane_diff_proj.build((None, 5, self.embed_dim))
        self.lane_sum_proj.build((None, 5, self.embed_dim))

        if self.use_sinkhorn_attention and self.sinkhorn_attn is not None:
            self.sinkhorn_attn.build([(None, 5, self.embed_dim), (None, 5, self.embed_dim)])
        elif self.use_cross_attention and self.cross_attn is not None:
            self.cross_attn.build((None, 5, self.embed_dim), (None, 5, self.embed_dim))

        super().build(input_shape)

    def call(
        self,
        inputs: tuple[tf.Tensor, ...] | list[tf.Tensor] | dict[str, tf.Tensor],
        training: bool | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if isinstance(inputs, dict):
            team_a = inputs["team_a"]
            team_b = inputs["team_b"]
            macro_clash = inputs.get("macro_clash", None)
        elif isinstance(inputs, (tuple, list)) and len(inputs) in (2, 3):
            team_a = inputs[0]
            team_b = inputs[1]
            macro_clash = inputs[2] if len(inputs) == 3 else None
        else:
            raise ValueError(f"SiameseTeamEncoder expects (team_a, team_b) or (team_a, team_b, macro_clash), got {inputs}")

        def _process_team(team: tf.Tensor):
            role_indices = tf.clip_by_value(tf.cast(tf.round(team[:, :, 1]), tf.int32), 0, 4)
            order = tf.argsort(role_indices, axis=1)
            ordered_team = tf.gather(team, order, batch_dims=1)

            c_ids = tf.clip_by_value(tf.cast(tf.round(ordered_team[:, :, 0]), tf.int32), 0, self.num_champions - 1)
            r_ids = tf.clip_by_value(tf.cast(tf.round(ordered_team[:, :, 1]), tf.int32), 0, 4)
            continuous_features = ordered_team[:, :, 2:]

            e_champ = self.champ_embed(c_ids)  # (batch, 5, champ_embed_dim)
            e_role = self.role_embed(r_ids)    # (batch, 5, role_embed_dim)

            if self.use_periodic_embeddings:
                if self.periodic_embed is None:
                    c_dim = continuous_features.shape[-1]
                    if c_dim is None:
                        c_dim = 12
                    self.periodic_embed = PeriodicNumericalEmbedding(
                        num_features=int(c_dim),
                        num_frequencies=self.num_frequencies,
                        trainable=True,
                        name="periodic_embed",
                    )
                cont_feat = self.periodic_embed(continuous_features)
            else:
                cont_feat = continuous_features

            entity_in = tf.concat([e_champ, e_role, cont_feat], axis=-1)
            h = self.phi2(self.phi1(entity_in))  # (batch, 5, embed_dim)

            if self.dropout is not None:
                h = self.dropout(h, training=training)

            # DeepSets permutation-invariant pool
            sum_pool = tf.reduce_sum(h, axis=1)  # (batch, embed_dim)
            if self.use_set_transformer and self.set_transformer is not None:
                pma_pool = self.set_transformer(h, training=training)  # (batch, 2 * embed_dim)
                pool = tf.concat([sum_pool, pma_pool], axis=-1)
            else:
                pool = sum_pool

            t = self.rho2(self.rho1(pool))  # (batch, latent_dim)

            # Bilinear champion synergy (FM pooling: 0.5 * ((sum v)^2 - sum v^2))
            sum_c = tf.reduce_sum(e_champ, axis=1)
            sum_sq_c = tf.reduce_sum(tf.square(e_champ), axis=1)
            synergy = 0.5 * (tf.square(sum_c) - sum_sq_c)
            syn_feat = self.synergy_proj(synergy)  # (batch, latent_dim)

            return h, t, syn_feat

        h_a, t_a_pool, syn_a = _process_team(team_a)
        h_b, t_b_pool, syn_b = _process_team(team_b)

        # 1. Lane matchup differences (Team B - Team A for each lane 0..4)
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

        # 3. Assemble Anti-Symmetric differences and Symmetric sums
        diff_components = [pool_diff, syn_diff, lane_diff_flat]
        sum_components = [pool_sum, syn_sum, lane_sum_flat]

        if macro_clash is not None:
            # Macro clash is strictly anti-symmetric: clash(B, A) == -clash(A, B)
            diff_components.append(macro_clash)

        if self.use_sinkhorn_attention and self.sinkhorn_attn is not None:
            attn_a, attn_b = self.sinkhorn_attn((h_a, h_b), training=training)
            attn_diff = attn_b - attn_a
            attn_sum = attn_a + attn_b
            attn_diff_flat = tf.reshape(attn_diff, [-1, 5 * self.embed_dim])
            attn_sum_flat = tf.reshape(attn_sum, [-1, 5 * self.embed_dim])
            diff_components.append(attn_diff_flat)
            sum_components.append(attn_sum_flat)
        elif self.use_cross_attention and self.cross_attn is not None:
            attn_b_to_a = self.cross_attn(h_b, h_a)
            attn_a_to_b = self.cross_attn(h_a, h_b)
            attn_diff = attn_b_to_a - attn_a_to_b
            attn_sum = attn_b_to_a + attn_a_to_b
            attn_diff_flat = tf.reshape(attn_diff, [-1, 5 * self.embed_dim])
            attn_sum_flat = tf.reshape(attn_sum, [-1, 5 * self.embed_dim])
            diff_components.append(attn_diff_flat)
            sum_components.append(attn_sum_flat)

        t_diff = tf.concat(diff_components, axis=-1, name="t_diff")
        t_sum = tf.concat(sum_components, axis=-1, name="t_sum")

        # 4. Team representations
        t_a = tf.concat([t_a_pool, syn_a], axis=-1, name="t_a")
        t_b = tf.concat([t_b_pool, syn_b], axis=-1, name="t_b")

        return t_diff, t_sum, t_a, t_b

    def get_config(self) -> dict[str, Any]:
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
            "use_cross_attention": self.use_cross_attention,
            "use_set_transformer": self.use_set_transformer,
            "use_sinkhorn_attention": self.use_sinkhorn_attention,
            "use_periodic_embeddings": self.use_periodic_embeddings,
            "num_frequencies": self.num_frequencies,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="nnriot")
class AntiSymmetricSoftmax(tf.keras.layers.Layer):
    """Takes a scalar logit s and outputs 2-class softmax [-s, s], strictly anti-symmetric."""

    def call(self, s: tf.Tensor) -> tf.Tensor:
        neg_s = -s
        logits = tf.concat([neg_s, s], axis=-1)
        return tf.nn.softmax(logits, axis=-1)


@tf.keras.utils.register_keras_serializable(package="nnriot")
class DirectionalSideBias(tf.keras.layers.Layer):
    """Additive scalar bias representing arena side advantage (e.g. Summoner's Rift Blue side).

    Decouples:
    - Pure draft interaction f_draft(A, B) where f(A, B) == -f(B, A)
    - Real-world map asymmetry b_side where logit = f_draft + b_side
    """

    def __init__(
        self,
        initial_bias: float = 0.0,
        trainable: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.initial_bias = float(initial_bias)
        self.bias_trainable = bool(trainable)

    def build(self, input_shape: Any = None) -> None:
        if self.built:
            return
        self.bias = self.add_weight(
            name="side_bias",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_bias),
            trainable=self.bias_trainable,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + self.bias

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "initial_bias": self.initial_bias,
            "trainable": self.bias_trainable,
        })
        return config
