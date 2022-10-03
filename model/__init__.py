import tensorflow as tf


def build_model() -> tf.keras.Model:
    backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False,
    )
    backbone.trainable = False

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3), name="image"),
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                units=512,
                activation="swish",
            ),
            tf.keras.layers.Dense(
                units=256,
                activation="swish",
            ),
            tf.keras.layers.Dense(
                units=64,
                activation="swish",
            ),
            tf.keras.layers.Dense(units=4, activation="sigmoid", name="bbox"),
        ]
    )

    return model
