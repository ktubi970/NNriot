"""Equivariant Siamese DeepSets multi-task architecture for League of Legends match predictions.

Guarantees exact mathematical team-swap anti-symmetry and permutation invariance:
- P(winner=1 | Team B, Team A) == 1 - P(winner=1 | Team A, Team B)
- kill_handicap(Team B, Team A) == -kill_handicap(Team A, Team B)
- team_a_kills(Team B, Team A) == team_b_kills(Team A, Team B)
- total_kills(Team B, Team A) == total_kills(Team A, Team B)
- first_*(Team B, Team A) swaps Team A (class 0) and Team B (class 1) probabilities
"""
from __future__ import annotations

import tensorflow as tf
from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
from core.generate_graph import LOSS_PER_HEAD, LOSS_WEIGHTS, METRICS_PER_HEAD


@tf.keras.utils.register_keras_serializable(package="nnriot")
class ThreeClassEquivariantSoftmax(tf.keras.layers.Layer):
    """Combines team A logit, team B logit, and neither logit into a 3-class softmax."""

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        # inputs: [l_a, l_b, l_neither]
        logits = tf.concat(inputs, axis=-1)
        return tf.nn.softmax(logits, axis=-1)


def build_equivariant_deepsets_model(
    player_feature_dim: int = 8,
    embed_dim: int = 32,
    latent_dim: int = 64,
    dropout_rate: float = 0.15,
    l2_reg: float = 1e-4,
    learning_rate: float = 0.0008,
) -> tf.keras.Model:
    """Build the equivariant Siamese DeepSets multi-task model.
    
    Inputs:
        {"team_a": Input(shape=(5, player_feature_dim)),
         "team_b": Input(shape=(5, player_feature_dim))}
         
    Outputs:
        Dictionary of 22 named head tensors matching ALL_LABEL_KEYS.
    """
    input_a = tf.keras.Input(shape=(5, player_feature_dim), name="team_a")
    input_b = tf.keras.Input(shape=(5, player_feature_dim), name="team_b")

    encoder = SiameseTeamEncoder(
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        name="siamese_team_encoder",
    )
    t_diff, t_sum, t_a, t_b = encoder((input_a, input_b))

    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    drop = tf.keras.layers.Dropout(dropout_rate, name="head_dropout") if dropout_rate > 0 else None
    t_diff_in = drop(t_diff) if drop is not None else t_diff
    t_sum_in = drop(t_sum) if drop is not None else t_sum

    outputs: dict[str, tf.Tensor] = {}

    # 1. Directional anti-symmetric 2-class softmax heads: winner & team_b_kill_lead
    # Logit s(t_diff) has no bias so s(-t_diff) == -s(t_diff)
    for name in ("winner", "team_b_kill_lead"):
        s = tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=reg, name=f"{name}_logit")(t_diff_in)
        outputs[name] = AntiSymmetricSoftmax(name=name)(s)

    # 2. Directional anti-symmetric linear regression: kill_handicap (Team A kills - Team B kills)
    outputs["kill_handicap"] = tf.keras.layers.Dense(
        1, use_bias=False, activation="linear", kernel_regularizer=reg, name="kill_handicap"
    )(t_diff_in)

    # 3. Individual team kills: team_a_kills and team_b_kills
    # Using a single shared linear layer so team_a_kills(B, A) == team_b_kills(A, B)
    team_kill_layer = tf.keras.layers.Dense(1, activation="linear", kernel_regularizer=reg, name="team_kills_shared")
    outputs["team_a_kills"] = tf.keras.layers.Activation("linear", name="team_a_kills")(team_kill_layer(t_a))
    outputs["team_b_kills"] = tf.keras.layers.Activation("linear", name="team_b_kills")(team_kill_layer(t_b))

    # 4. Symmetric aggregate regression heads (depend on game pace / sum of both teams)
    for name in ("total_kills", "total_barons", "total_dragons", "total_towers"):
        h = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg, name=f"{name}_hidden")(t_sum_in)
        outputs[name] = tf.keras.layers.Dense(1, activation="linear", kernel_regularizer=reg, name=name)(h)

    # 5. Symmetric binary sigmoid heads (depend on aggregate game state)
    for name in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon"):
        h = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg, name=f"{name}_hidden")(t_sum_in)
        outputs[name] = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name=name)(h)

    # 6. 3-class first objective heads (0=Team A, 1=Team B, 2=Neither)
    three_class_heads = (
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

    model = tf.keras.Model(inputs={"team_a": input_a, "team_b": input_b}, outputs=outputs, name="NNriot_EquivariantDeepSets")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=LOSS_PER_HEAD,
        loss_weights=LOSS_WEIGHTS,
        metrics=METRICS_PER_HEAD,
    )

    return model
