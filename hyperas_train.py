import datetime
from pathlib import Path

import tensorflow as tf
from hyperas import optim
from hyperopt import tpe, Trials
from hyperopt.pyll.stochastic import lognormal
from keras.optimizers import Adam, RMSprop
from callbacks import TensorboardReportImages
from metrics import IOU
from train import make_dataset, make_dataset_split

from hyperas.distributions import choice, quniform, normal

OUT_ROOT = Path("../out/")
TENSORBOARD_LOG = Path("../logs/fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def data() -> dict:
    dataset_root = Path("../dataset/")
    dataset_config = {
        "batch_size": None,
    }

    return {
        split: make_dataset_split(dataset_root / split, dataset_config)
        for split in ["train", "val", "test"]
    }


def build_model(dataset: dict) -> dict:
    inputs = (tf.keras.layers.Input(shape=(224, 224, 3), name="image"),)

    backbone = tf.keras.applications.convnext.ConvNeXtTiny(
        include_top=False,
    )(inputs)
    backbone.trainable = True

    out = tf.keras.layers.GlobalAveragePooling2D()(backbone)
    out = out[0]

    num_layers = int({{quniform(1, 4, 1)}})
    for ind in range(num_layers):
        out = tf.keras.layers.Dense(
            units={{choice[512, 256, 64, 16]}},
            activation={{choice(["relu", "swish"])}}
        )

    out = tf.keras.layers.Dense(units=4, activation="sigmoid", name="bbox")(out)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    lr = {{choice([1e-3, 1e-4, 1e-5])}}
    optimizer = {{choice([Adam, RMSprop])}}

    optimizer = optimizer(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps={{normal(5600, 1400)}},
            decay_rate={{lognormal(0.1, 0.2)}},
        )
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "bbox": "mse",
        },
        metrics=[
            IOU(),
        ],
    )

    model.fit(
        dataset["train"],
        validation_data=dataset["val"],
        batch_size={{choice([16, 32])}},
        epochs=300,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_iou",
                mode="max",
                patience=100,
                min_delta=0.01,
            )
        ],
        verbose=0,
    )

    results = model.evaluate(
        dataset["test"],
        batch_size=64,
        verbose=0,
    )

    test_iou = results[1]

    return {
        "loss": -test_iou,
        "status": "ok",
        "model": model,
    }


def main():
    best_pms, best_model = optim.minimize(
        model=build_model,
        data=data,
        algo=tpe.suggest,
        max_evals=20,
        trials=Trials(),
    )

    print(f"Best pms: {best_pms}")
    tf.keras.models.save_model(best_model, str(OUT_ROOT / "saved_models" / "model"))


if __name__ == "__main__":
    main()
