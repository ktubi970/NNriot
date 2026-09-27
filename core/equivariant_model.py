"""Equivariant Siamese DeepSets multi-task architecture for League of Legends match predictions.

Guarantees exact mathematical team-swap anti-symmetry and permutation invariance:
- P(winner=1 | Team B, Team A) == 1 - P(winner=1 | Team A, Team B)
- kill_handicap(Team B, Team A) == -kill_handicap(Team A, Team B)
- team_a_kills(Team B, Team A) == team_b_kills(Team A, Team B)
- total_kills(Team B, Team A) == total_kills(Team A, Team B)
- first_*(Team B, Team A) swaps Team A (class 0) and Team B (class 1) probabilities
- winner_uncertainty(Team B, Team A) == winner_uncertainty(Team A, Team B)
"""
from __future__ import annotations

from typing import Any
import tensorflow as tf
from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
from core.generate_graph import (
    LOSS_PER_HEAD,
    LOSS_WEIGHTS,
    METRICS_PER_HEAD,
    PRUNED_LABEL_KEYS,
    PRUNED_LOSS_PER_HEAD,
    PRUNED_LOSS_WEIGHTS,
    PRUNED_METRICS_PER_HEAD,
)


@tf.keras.utils.register_keras_serializable(package="nnriot")
class ThreeClassEquivariantSoftmax(tf.keras.layers.Layer):
    """Combines team A logit, team B logit, and neither logit into a 3-class softmax."""

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        # inputs: [l_a, l_b, l_neither]
        logits = tf.concat(inputs, axis=-1)
        return tf.nn.softmax(logits, axis=-1)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="nnriot")
class SkewSymmetricBilinearDense(tf.keras.layers.Layer):
    """Exact skew-symmetric bilinear interaction between two team latent representations.

    Given inputs (t_a, t_b) each of shape (batch, dim), computes:
        s = t_b^T (W - W^T) t_a

    Since M = W - W^T is strictly anti-symmetric (M^T = -M), this guarantees:
        s(t_b, t_a) = t_a^T M t_b = (t_b^T M^T t_a)^T = - t_b^T M t_a = -s(t_a, t_b)
    to machine precision.
    Additionally, for mirror matchups (t_a == t_b), s(t_a, t_a) == 0.0 identically.
    """

    def __init__(
        self,
        kernel_regularizer: tf.keras.regularizers.Regularizer | None = None,
        initializer: str = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape: Any = None) -> None:
        if self.built:
            return
        if (
            isinstance(input_shape, (tuple, list))
            and len(input_shape) >= 1
            and isinstance(input_shape[0], (tuple, list, tf.TensorShape))
        ):
            dim = int(input_shape[0][-1])
        elif hasattr(input_shape, "__getitem__"):
            dim = int(input_shape[-1])
        else:
            raise ValueError(f"SkewSymmetricBilinearDense cannot determine dimension from input_shape={input_shape}")

        self.kernel = self.add_weight(
            name="kernel",
            shape=(dim, dim),
            initializer=self.initializer,
            regularizer=self.kernel_regularizer,
            trainable=self.trainable,
        )
        super().build(input_shape)

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor] | tf.Tensor,
        t_b: tf.Tensor | None = None,
    ) -> tf.Tensor:
        if t_b is not None:
            t_a = inputs
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            t_a, t_b = inputs[0], inputs[1]
        else:
            raise ValueError("SkewSymmetricBilinearDense expects (t_a, t_b) as a 2-tuple or two positional arguments.")

        m = self.kernel - tf.transpose(self.kernel)
        tb_m = tf.matmul(t_b, m)
        s = tf.reduce_sum(tb_m * t_a, axis=-1, keepdims=True)
        return s

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "initializer": tf.keras.initializers.serialize(self.initializer),
        })
        return config


@tf.keras.utils.register_keras_serializable(package="nnriot")
class EvidentialDirichletSoftmax(tf.keras.layers.Layer):
    """Dirichlet evidential uncertainty head for anti-symmetric 2-class predictions.

    Based on Sensoy et al. (NeurIPS 2018) "Evidential Deep Learning to Quantify Classification Uncertainty".

    Given scalar anti-symmetric logit s (where +s favors Team B, -s favors Team A):
    - Evidence: e_a = softplus(-s), e_b = softplus(+s) >= 0
    - Dirichlet concentration: alpha_a = e_a + 1, alpha_b = e_b + 1
    - Dirichlet strength (precision): S = alpha_a + alpha_b = e_a + e_b + 2 >= 2
    - Expected probabilities: p_a = alpha_a / S, p_b = alpha_b / S
    - Epistemic uncertainty (vacuity): u = 2 / S in (0, 1]

    Mathematical Invariants:
    - P(Team B | -s) == P(Team A | s) == 1 - P(Team B | s) (exact anti-symmetry)
    - u(-s) == u(s) (exact symmetry / invariance under team swap)
    - 0 < u <= 1
    """

    def __init__(
        self,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.return_dict = bool(return_dict)

    def call(
        self,
        s: tf.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor] | dict[str, tf.Tensor]:
        neg_s = -s
        e_a = tf.math.softplus(neg_s)
        e_b = tf.math.softplus(s)

        alpha_a = e_a + 1.0
        alpha_b = e_b + 1.0
        s_total = alpha_a + alpha_b

        p_a = alpha_a / s_total
        p_b = alpha_b / s_total
        p = tf.concat([p_a, p_b], axis=-1)
        u = 2.0 / s_total

        should_return_dict = self.return_dict if return_dict is None else return_dict
        if should_return_dict:
            return {"winner": p, "winner_uncertainty": u}
        return p, u

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "return_dict": self.return_dict,
        })
        return config


def build_equivariant_deepsets_model(
    player_feature_dim: int = 14,
    macro_feature_dim: int = 15,
    embed_dim: int = 32,
    latent_dim: int = 64,
    dropout_rate: float = 0.15,
    l2_reg: float = 1e-4,
    learning_rate: float = 0.0008,
    use_cross_attention: bool = False,
    include_macro_clash: bool = False,
    use_evidential: bool = True,
    pruned_heads: bool = True,
    use_bilinear: bool = True,
) -> tf.keras.Model:
    """Build the equivariant Siamese DeepSets multi-task model.

    Inputs:
        {"team_a": Input(shape=(5, player_feature_dim)),
         "team_b": Input(shape=(5, player_feature_dim))}
        and optionally "macro_clash": Input(shape=(macro_feature_dim,))

    Outputs:
        When pruned_heads=True (default):
            Dictionary of 7 core heads (from PRUNED_LABEL_KEYS) + "winner_uncertainty" if use_evidential=True.
        When pruned_heads=False:
            Dictionary of 22 named head tensors matching ALL_LABEL_KEYS + "winner_uncertainty" if use_evidential=True.
    """
    input_a = tf.keras.Input(shape=(5, player_feature_dim), name="team_a")
    input_b = tf.keras.Input(shape=(5, player_feature_dim), name="team_b")

    if include_macro_clash and macro_feature_dim > 0:
        input_macro = tf.keras.Input(shape=(macro_feature_dim,), name="macro_clash")
        inputs = {"team_a": input_a, "team_b": input_b, "macro_clash": input_macro}
        encoder_inputs = (input_a, input_b, input_macro)
    else:
        inputs = {"team_a": input_a, "team_b": input_b}
        encoder_inputs = (input_a, input_b)

    encoder = SiameseTeamEncoder(
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        use_cross_attention=use_cross_attention,
        name="siamese_team_encoder",
    )
    t_diff, t_sum, t_a, t_b = encoder(encoder_inputs)

    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    drop = tf.keras.layers.Dropout(dropout_rate, name="head_dropout") if dropout_rate > 0 else None
    t_diff_in = drop(t_diff) if drop is not None else t_diff
    t_sum_in = drop(t_sum) if drop is not None else t_sum

    outputs: dict[str, tf.Tensor] = {}

    # 1. Winner head (Directional anti-symmetric probability + optional Dirichlet epistemic uncertainty)
    if use_bilinear:
        winner_bilinear = SkewSymmetricBilinearDense(
            kernel_regularizer=reg, name="winner_bilinear"
        )((t_a, t_b))
        winner_linear = tf.keras.layers.Dense(
            1, use_bias=False, kernel_regularizer=reg, name="winner_linear_logit"
        )(t_diff_in)
        s_winner = tf.keras.layers.Add(name="winner_logit")([winner_bilinear, winner_linear])
    else:
        s_winner = tf.keras.layers.Dense(
            1, use_bias=False, kernel_regularizer=reg, name="winner_logit"
        )(t_diff_in)

    if use_evidential:
        p_winner, u_winner = EvidentialDirichletSoftmax(name="winner_evidential")(s_winner)
        outputs["winner"] = tf.keras.layers.Activation("linear", name="winner")(p_winner)
        outputs["winner_uncertainty"] = tf.keras.layers.Activation("linear", name="winner_uncertainty")(u_winner)
    else:
        outputs["winner"] = AntiSymmetricSoftmax(name="winner")(s_winner)

    # 2. Team B kill lead (Directional anti-symmetric 2-class softmax)
    s_lead = tf.keras.layers.Dense(
        1, use_bias=False, kernel_regularizer=reg, name="team_b_kill_lead_logit"
    )(t_diff_in)
    outputs["team_b_kill_lead"] = AntiSymmetricSoftmax(name="team_b_kill_lead")(s_lead)

    # 3. Directional anti-symmetric linear regression: kill_handicap (Team A kills - Team B kills)
    if use_bilinear:
        handicap_bilinear = SkewSymmetricBilinearDense(
            kernel_regularizer=reg, name="kill_handicap_bilinear"
        )((t_a, t_b))
        handicap_linear = tf.keras.layers.Dense(
            1, use_bias=False, activation="linear", kernel_regularizer=reg, name="kill_handicap_linear"
        )(t_diff_in)
        outputs["kill_handicap"] = tf.keras.layers.Add(name="kill_handicap")([handicap_bilinear, handicap_linear])
    else:
        outputs["kill_handicap"] = tf.keras.layers.Dense(
            1, use_bias=False, activation="linear", kernel_regularizer=reg, name="kill_handicap"
        )(t_diff_in)

    # 4. Individual team kills: team_a_kills and team_b_kills (shared linear layer)
    team_kill_layer = tf.keras.layers.Dense(1, activation="linear", kernel_regularizer=reg, name="team_kills_shared")
    outputs["team_a_kills"] = tf.keras.layers.Activation("linear", name="team_a_kills")(team_kill_layer(t_a))
    outputs["team_b_kills"] = tf.keras.layers.Activation("linear", name="team_b_kills")(team_kill_layer(t_b))

    # 5. Symmetric aggregate regression heads
    total_heads = ("total_kills",) if pruned_heads else ("total_kills", "total_barons", "total_dragons", "total_towers")
    for name in total_heads:
        h = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg, name=f"{name}_hidden")(t_sum_in)
        outputs[name] = tf.keras.layers.Dense(1, activation="linear", kernel_regularizer=reg, name=name)(h)

    # 6. Symmetric binary sigmoid heads (only if not pruned)
    if not pruned_heads:
        for name in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon"):
            h = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg, name=f"{name}_hidden")(t_sum_in)
            outputs[name] = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name=name)(h)

    # 7. 3-class first objective heads
    three_class_heads = ("first_blood",) if pruned_heads else (
        "first_blood", "first_baron", "first_inhibitor", "first_tower",
        "first_to_5_kills", "first_to_10_kills", "first_to_15_kills", "first_to_20_kills"
    )
    for name in three_class_heads:
        team_logit_dense = tf.keras.layers.Dense(1, use_bias=True, kernel_regularizer=reg, name=f"{name}_team_logit")
        neither_logit_dense = tf.keras.layers.Dense(1, use_bias=True, kernel_regularizer=reg, name=f"{name}_neither_logit")

        l_a = team_logit_dense(t_a)
        l_b = team_logit_dense(t_b)
        l_neither = neither_logit_dense(t_sum_in)

        outputs[name] = ThreeClassEquivariantSoftmax(name=name)([l_a, l_b, l_neither])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NNriot_EquivariantDeepSets")

    # Compile with appropriate losses/weights/metrics for active heads
    if pruned_heads:
        loss_dict = {k: v for k, v in PRUNED_LOSS_PER_HEAD.items() if k in outputs}
        loss_weights_dict = {k: v for k, v in PRUNED_LOSS_WEIGHTS.items() if k in outputs}
        metrics_dict = {k: v for k, v in PRUNED_METRICS_PER_HEAD.items() if k in outputs}
    else:
        loss_dict = {k: v for k, v in LOSS_PER_HEAD.items() if k in outputs}
        loss_weights_dict = {k: v for k, v in LOSS_WEIGHTS.items() if k in outputs}
        metrics_dict = {k: v for k, v in METRICS_PER_HEAD.items() if k in outputs}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=metrics_dict,
    )

    return model
