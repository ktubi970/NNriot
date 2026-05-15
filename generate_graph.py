import os

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import warnings
import logging

# Suppress all warnings and internal TF logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# Output head name -> loss function (Keras strings).
# Keys must match feature_labels.ALL_LABEL_KEYS exactly (22 heads = 18 original
# + 4 timeline first_to_N_kills).
LOSS_PER_HEAD: dict[str, str] = {
    "winner":               "categorical_crossentropy",
    "team_b_kill_lead":         "categorical_crossentropy",
    "first_blood":          "categorical_crossentropy",
    "first_baron":          "categorical_crossentropy",
    "first_inhibitor":      "categorical_crossentropy",
    "first_tower":          "categorical_crossentropy",
    "kills_odd":            "binary_crossentropy",
    "both_baron":           "binary_crossentropy",
    "both_inhibitor":       "binary_crossentropy",
    "both_dragon":          "binary_crossentropy",
    "elder_dragon":         "binary_crossentropy",
    "kill_handicap":        "mse",
    "total_kills":          "mse",
    "team_a_kills":         "mse",
    "team_b_kills":         "mse",
    "total_barons":         "mse",
    "total_dragons":        "mse",
    "total_towers":         "mse",
    # Timeline-derived 3-class softmax heads (0=blue first, 1=red first, 2=neither)
    "first_to_5_kills":     "categorical_crossentropy",
    "first_to_10_kills":    "categorical_crossentropy",
    "first_to_15_kills":    "categorical_crossentropy",
    "first_to_20_kills":    "categorical_crossentropy",
}

# Output head name -> loss weight (starting point per MULTI_OUTPUT_MODEL_PLAN.md §5).
# Regression heads run at 1.0 because their targets are normalized to ~unit
# variance in continuous_trainer._build_targets (see REGRESSION_STATS).
LOSS_WEIGHTS: dict[str, float] = {
    "winner":            1.0,
    "team_b_kill_lead":  1.0,
    "kills_odd":         0.5,
    "first_blood":       0.7,
    "first_baron":       0.7,
    "first_inhibitor":   0.7,
    "first_tower":       0.5,
    "both_baron":        0.4,
    "both_inhibitor":    0.4,
    "both_dragon":       0.4,
    "elder_dragon":      0.3,
    "total_kills":       1.0,
    "total_barons":      1.0,
    "total_dragons":     1.0,
    "total_towers":      1.0,
    "team_a_kills":      1.0,
    "team_b_kills":      1.0,
    "kill_handicap":     1.0,
    # Timeline kill thresholds — weights decay with rarity (most matches
    # never hit 15 or 20 team kills, so those targets are heavily class-2).
    "first_to_5_kills":  0.7,
    "first_to_10_kills": 0.7,
    "first_to_15_kills": 0.5,
    "first_to_20_kills": 0.3,
}

# Output head name -> list of Keras metric strings (for logging during training).
METRICS_PER_HEAD: dict[str, list[str]] = {
    "winner":               ["accuracy"],
    "team_b_kill_lead":         ["accuracy"],
    "first_blood":          ["accuracy"],
    "first_baron":          ["accuracy"],
    "first_inhibitor":      ["accuracy"],
    "first_tower":          ["accuracy"],
    "kills_odd":            ["accuracy"],
    "both_baron":           ["accuracy"],
    "both_inhibitor":       ["accuracy"],
    "both_dragon":          ["accuracy"],
    "elder_dragon":         ["accuracy"],
    "kill_handicap":        ["mae"],
    "total_kills":          ["mae"],
    "team_a_kills":         ["mae"],
    "team_b_kills":         ["mae"],
    "total_barons":         ["mae"],
    "total_dragons":        ["mae"],
    "total_towers":         ["mae"],
    "first_to_5_kills":     ["accuracy"],
    "first_to_10_kills":    ["accuracy"],
    "first_to_15_kills":    ["accuracy"],
    "first_to_20_kills":    ["accuracy"],
}


def build_keras_model(input_dim=100000):
    inputs = tf.keras.Input(shape=(input_dim,), name="Input")

    # Projection
    x = tf.keras.layers.Dense(1024, activation="relu", name="Projection_Dense")(inputs)

    # ResBlock 1
    res1 = tf.keras.layers.Dense(1024, activation="relu", name="ResBlock1_layer1")(x)
    res1 = tf.keras.layers.Dense(1024, activation=None, name="ResBlock1_layer2")(res1)
    x = tf.keras.layers.Add(name="ResBlock1_add")([x, res1])
    x = tf.keras.layers.Activation("relu", name="ResBlock1_relu")(x)

    # ResBlock 2
    res2 = tf.keras.layers.Dense(1024, activation="relu", name="ResBlock2_layer1")(x)
    res2 = tf.keras.layers.Dense(1024, activation=None, name="ResBlock2_layer2")(res2)
    x = tf.keras.layers.Add(name="ResBlock2_add")([x, res2])
    x = tf.keras.layers.Activation("relu", name="ResBlock2_relu")(x)

    # Bottleneck
    x = tf.keras.layers.Dense(512, activation="relu", name="Dense512")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="Dense128")(x)

    # Output
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="Output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NNriot_ResMLP")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def _head(name: str, units: int, activation: str | None, source, hidden_size: int = 64):
    """Build a per-head capacity layer (Dense hidden_size ReLU) before the final output Dense."""
    h = tf.keras.layers.Dense(hidden_size, activation="relu", name=f"{name}_hidden")(source)
    return tf.keras.layers.Dense(units, activation=activation, name=name)(h)


def build_multi_output_model(input_dim: int = 20000, dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Build the 22-output multi-task model.

    Shared trunk (identical to build_keras_model):
      Input -> Dense 1024 -> ResBlock x 2 -> Dense 512 -> Dense 128 (embedding)

    Per-task heads off the 128-dim embedding (22 named outputs = 18 original
    + 4 timeline first_to_N_kills). Each head has a Dense(64, relu)
    "*_hidden" layer before its final output Dense, giving each head a bit
    of per-task capacity.

    Dropout (default 0.3) is applied after each trunk dense layer
    (Projection, ResBlock1_relu, ResBlock2_relu, Dense512) — 4 Dropout
    layers total. No dropout on the embedding or on heads.

    See MULTI_OUTPUT_MODEL_PLAN.md section 4 for the head spec.
    """
    from feature_labels import ALL_LABEL_KEYS
    assert set(LOSS_PER_HEAD) == set(LOSS_WEIGHTS) == set(METRICS_PER_HEAD) == set(ALL_LABEL_KEYS), \
        "LOSS_PER_HEAD / LOSS_WEIGHTS / METRICS_PER_HEAD must cover exactly ALL_LABEL_KEYS"

    inputs = tf.keras.Input(shape=(input_dim,), name="Input")

    # Projection (1024)
    x = tf.keras.layers.Dense(1024, activation="relu", name="Projection_Dense")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate, name="Projection_Dropout")(x)

    # ResBlock 1
    res1 = tf.keras.layers.Dense(1024, activation="relu", name="ResBlock1_layer1")(x)
    res1 = tf.keras.layers.Dense(1024, activation=None, name="ResBlock1_layer2")(res1)
    x = tf.keras.layers.Add(name="ResBlock1_add")([x, res1])
    x = tf.keras.layers.Activation("relu", name="ResBlock1_relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="ResBlock1_Dropout")(x)

    # ResBlock 2
    res2 = tf.keras.layers.Dense(1024, activation="relu", name="ResBlock2_layer1")(x)
    res2 = tf.keras.layers.Dense(1024, activation=None, name="ResBlock2_layer2")(res2)
    x = tf.keras.layers.Add(name="ResBlock2_add")([x, res2])
    x = tf.keras.layers.Activation("relu", name="ResBlock2_relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="ResBlock2_Dropout")(x)

    # Bottleneck
    x = tf.keras.layers.Dense(512, activation="relu", name="Dense512")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="Dense512_Dropout")(x)
    embedding = tf.keras.layers.Dense(128, activation="relu", name="embedding")(x)

    # Per-head outputs (each gains a Dense(64, relu) *_hidden layer via _head)
    outputs = {}
    # 2-class softmax heads
    for name in ("winner", "team_b_kill_lead"):
        outputs[name] = _head(name=name, units=2, activation="softmax", source=embedding)
    # 3-class softmax heads (first_* event ownership + first-to-N kill thresholds)
    for name in ("first_blood", "first_baron", "first_inhibitor", "first_tower",
                 "first_to_5_kills", "first_to_10_kills",
                 "first_to_15_kills", "first_to_20_kills"):
        outputs[name] = _head(name=name, units=3, activation="softmax", source=embedding)
    # Binary sigmoid heads
    for name in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon"):
        outputs[name] = _head(name=name, units=1, activation="sigmoid", source=embedding)
    # Regression heads
    for name in ("kill_handicap", "total_kills", "team_a_kills", "team_b_kills",
                 "total_barons", "total_dragons", "total_towers"):
        outputs[name] = _head(name=name, units=1, activation="linear", source=embedding)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NNriot_MultiHead")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=LOSS_PER_HEAD,
        loss_weights=LOSS_WEIGHTS,
        metrics=METRICS_PER_HEAD,
    )
    return model


if __name__ == "__main__":
    # Sanity build of the current multi-output model — does NOT save a checkpoint
    # (training is handled by continuous_trainer.py). The old code saved a legacy
    # model.keras that would shadow the real model_v2.keras checkpoint.
    model = build_multi_output_model()
    model.summary()
    print(f"Multi-output model has {len(model.output_names)} heads.")
