import tensorflow as tf


def build_model() -> tf.keras.Model:
    inputs = (tf.keras.layers.Input(shape=(224, 224, 3), name="image"),)

    backbone = tf.keras.applications.convnext.ConvNeXtTiny(
        include_top=False,
    )(inputs)
    backbone.trainable = True

    out = tf.keras.layers.GlobalAveragePooling2D()(backbone)

    out = tf.keras.layers.Dense(
        units=512,
        activation="swish",
    )(out)
    out = tf.keras.layers.Dropout(0.3)(out)
    out = tf.keras.layers.Dense(
        units=256,
        activation="swish",
    )(out)
    out = tf.keras.layers.Dropout(0.3)(out)
    out = tf.keras.layers.Dense(
        units=64,
        activation="swish",
    )(out)
    out = tf.keras.layers.Dense(units=4, activation="sigmoid", name="bbox")(out)

    return tf.keras.Model(inputs=inputs, outputs=out)
