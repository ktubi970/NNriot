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


from core.feature_labels import ALL_LABEL_KEYS, PRUNED_LABEL_KEYS

# Re-export PRUNED_LABEL_KEYS for generate_graph consumers
__all__ = [
    "LOSS_PER_HEAD",
    "LOSS_WEIGHTS",
    "METRICS_PER_HEAD",
    "PRUNED_LABEL_KEYS",
    "PRUNED_LOSS_PER_HEAD",
    "PRUNED_LOSS_WEIGHTS",
    "PRUNED_METRICS_PER_HEAD",
    "HomoscedasticLossLayer",
    "build_keras_model",
    "build_multi_output_model",
    "build_draft_dense_model",
    "build_enriched_model",
    "build_draft_multi_output_model",
]

# Pruned core 7 heads loss function mappings
PRUNED_LOSS_PER_HEAD: dict[str, str] = {
    "winner":               "categorical_crossentropy",
    "team_b_kill_lead":     "categorical_crossentropy",
    "kill_handicap":        "mse",
    "first_blood":          "categorical_crossentropy",
    "total_kills":          "mse",
    "team_a_kills":         "mse",
    "team_b_kills":         "mse",
}

# Pruned core 7 heads baseline loss weights
PRUNED_LOSS_WEIGHTS: dict[str, float] = {
    "winner":            1.0,
    "team_b_kill_lead":  1.0,
    "kill_handicap":     1.0,
    "first_blood":       0.7,
    "total_kills":       1.0,
    "team_a_kills":      1.0,
    "team_b_kills":      1.0,
}

# Pruned core 7 heads metrics
PRUNED_METRICS_PER_HEAD: dict[str, list[str]] = {
    "winner":               ["accuracy"],
    "team_b_kill_lead":     ["accuracy"],
    "kill_handicap":        ["mae"],
    "first_blood":          ["accuracy"],
    "total_kills":          ["mae"],
    "team_a_kills":         ["mae"],
    "team_b_kills":         ["mae"],
}


@tf.keras.utils.register_keras_serializable(package="nnriot")
class HomoscedasticLossLayer(tf.keras.layers.Layer):
    """Learned homoscedastic uncertainty weighting for multi-task losses (Kendall et al. CVPR 2018).

    L = sum_t ( exp(-s_t) * L_t + 0.5 * s_t ) where s_t = log(sigma_t^2).
    """

    def __init__(self, task_names: list[str] | tuple[str, ...], **kwargs):
        super().__init__(**kwargs)
        self.task_names = [str(name) for name in task_names]
        self.build(None)

    def build(self, input_shape=None):
        if not hasattr(self, "log_vars") or self.log_vars is None:
            self.log_vars = self.add_weight(
                name="log_vars",
                shape=(len(self.task_names),),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
        super().build(input_shape)

    def call(self, task_losses: dict[str, tf.Tensor] | list[tf.Tensor] | tuple[tf.Tensor, ...]) -> tf.Tensor:
        total = tf.constant(0.0, dtype=tf.float32)
        if isinstance(task_losses, dict):
            for idx, name in enumerate(self.task_names):
                loss_t = tf.cast(task_losses[name], tf.float32)
                loss_t = tf.reduce_mean(loss_t)
                s_t = self.log_vars[idx]
                precision = tf.exp(-s_t)
                total = total + precision * loss_t + 0.5 * s_t
        else:
            for idx, loss_t in enumerate(task_losses):
                loss_t = tf.cast(loss_t, tf.float32)
                loss_t = tf.reduce_mean(loss_t)
                s_t = self.log_vars[idx]
                precision = tf.exp(-s_t)
                total = total + precision * loss_t + 0.5 * s_t
        return total

    @property
    def loss_weights(self) -> dict[str, float]:
        """Return mapping of task name -> current precision weight exp(-s_t)."""
        if hasattr(self, "log_vars") and self.log_vars is not None:
            weights = tf.exp(-self.log_vars).numpy()
            return dict(zip(self.task_names, (float(w) for w in weights)))
        return {name: 1.0 for name in self.task_names}

    def get_config(self) -> dict:
        config = super().get_config()
        config["task_names"] = list(self.task_names)
        return config


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
# variance in training_utils._build_targets (see REGRESSION_STATS).
LOSS_WEIGHTS: dict[str, float] = {
    "winner":            1.0,
    "team_b_kill_lead":  1.0,
    "kills_odd":         0.0,
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
    # Timeline kill thresholds. Weights are 0.0 while the match_timelines table
    # is empty — every record's label defaults to class 2 ("neither team
    # reached the threshold"), so these heads contribute pure noise to the
    # shared trunk's gradient. Re-enable (suggested: 0.7/0.7/0.5/0.3) once
    # NNRIOT_FETCH_TIMELINES=1 has been run long enough to populate the
    # timeline column on a meaningful fraction of records.
    "first_to_5_kills":  0.0,
    "first_to_10_kills": 0.0,
    "first_to_15_kills": 0.0,
    "first_to_20_kills": 0.0,
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
    from core.feature_labels import ALL_LABEL_KEYS
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


def build_draft_dense_model(input_dim: int | None = None, dropout_rate: float = 0.2) -> tf.keras.Model:
    """Compact, focused MLP specifically for dense draft features."""
    if input_dim is None:
        from core.draft_features import DENSE_FEATURE_DIM
        input_dim = DENSE_FEATURE_DIM

    inputs = tf.keras.Input(shape=(input_dim,), name="Input")
    x = tf.keras.layers.Dense(512, activation="relu", name="Draft_Dense1")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="Draft_Dense2")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="Draft_Dense3")(x)
    outputs = {"winner": tf.keras.layers.Dense(2, activation="softmax", name="winner")(x)}

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NNriot_DraftDense")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"winner": "categorical_crossentropy"},
        metrics={"winner": ["accuracy"]},
    )
    return model


def build_enriched_model(
    input_dim: int | None = None,
    dropout_rate: float = 0.2,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """Compact regularized model for enriched features (draft + point-in-time player priors)."""
    if input_dim is None:
        from core.enriched_features import ENRICHED_DENSE_DIM
        input_dim = ENRICHED_DENSE_DIM

    inputs = tf.keras.Input(shape=(input_dim,), name="Input")
    x = tf.keras.layers.Dense(
        128, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="Enriched_Dense1",
    )(inputs)
    x = tf.keras.layers.Dropout(dropout_rate, name="Enriched_Drop1")(x)
    x = tf.keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="Enriched_Dense2",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="Enriched_Drop2")(x)
    embedding = tf.keras.layers.Dense(32, activation="relu", name="embedding")(x)

    outputs = {"winner": tf.keras.layers.Dense(2, activation="softmax", name="winner")(embedding)}

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NNriot_Enriched")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"winner": "categorical_crossentropy"},
        metrics={"winner": ["accuracy"]},
    )
    return model



def build_draft_multi_output_model(input_dim: int | None = None, dropout_rate: float = 0.2) -> tf.keras.Model:
    """Compact multi-output model for dense draft features (~350k params vs 25M)."""
    if input_dim is None:
        from core.draft_features import DENSE_FEATURE_DIM
        input_dim = DENSE_FEATURE_DIM

    inputs = tf.keras.Input(shape=(input_dim,), name="Input")
    x = tf.keras.layers.Dense(256, activation="relu", name="Draft_Shared1")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="Draft_Shared2")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    embedding = tf.keras.layers.Dense(64, activation="relu", name="embedding")(x)

    outputs = {}
    for name in ("winner", "team_b_kill_lead"):
        outputs[name] = _head(name=name, units=2, activation="softmax", source=embedding, hidden_size=32)
    for name in ("first_blood", "first_baron", "first_inhibitor", "first_tower",
                 "first_to_5_kills", "first_to_10_kills", "first_to_15_kills", "first_to_20_kills"):
        outputs[name] = _head(name=name, units=3, activation="softmax", source=embedding, hidden_size=32)
    for name in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon"):
        outputs[name] = _head(name=name, units=1, activation="sigmoid", source=embedding, hidden_size=32)
    for name in ("kill_handicap", "total_kills", "team_a_kills", "team_b_kills",
                 "total_barons", "total_dragons", "total_towers"):
        outputs[name] = _head(name=name, units=1, activation="linear", source=embedding, hidden_size=32)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NNriot_DraftMultiOutput")
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
