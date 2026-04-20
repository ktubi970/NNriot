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


if __name__ == "__main__":
    model = build_keras_model()
    model.summary()
    model.save("model.keras")
    print("Keras model saved to model.keras")
