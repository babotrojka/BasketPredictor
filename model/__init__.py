import tensorflow as tf


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3), name="image"),
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=5,
                strides=1,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.MaxPool2D(
                pool_size=4,
            ),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                strides=2,
                padding="valid",
                activation="relu",
            ),
            tf.keras.layers.MaxPool2D(
                pool_size=2,
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=128,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                units=64,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                units=4,
                name="bbox",
            ),
        ]
    )

    return model
