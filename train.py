import datetime
from pathlib import Path

import tensorflow as tf

from callbacks import TensorboardReportImages
from dataset import load_dataset, prepare_dataset
from metrics import IOU
from model import build_model

DATASET_ROOT = Path("../dataset/")

OUT_ROOT = Path("../out/")
TENSORBOARD_LOG = Path("../logs/fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def make_dataset_split(split_root: Path, dataset_config: dict) -> tf.data.Dataset:
    dataset = load_dataset(split_root)
    dataset = prepare_dataset(dataset, batch_size=dataset_config["batch_size"])
    return dataset


def make_dataset(
    dataset_root: Path, dataset_config: dict
) -> dict[str, tf.data.Dataset]:
    return {
        split: make_dataset_split(dataset_root / split, dataset_config)
        for split in ["train", "val", "test"]
    }


def main():
    dataset_config = {
        "batch_size": 32,
    }

    train_config = {
        "epochs": 300,
        "learning_rate": 1e-3,
    }

    dataset = make_dataset(DATASET_ROOT, dataset_config)

    model = build_model()
    print(model.summary())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=train_config["learning_rate"],
                decay_steps=5600,
                decay_rate=0.1,
            )
        ),
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
        epochs=train_config["epochs"],
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                TENSORBOARD_LOG,
            ),
            TensorboardReportImages(
                TENSORBOARD_LOG,
                dataset["val"].take(1),
            ),
        ],
    )

    model.evaluate(
        dataset["test"],
        batch_size=64,
        callbacks=[
            TensorboardReportImages(
                TENSORBOARD_LOG,
                dataset["test"],
            )
        ],
    )

    tf.keras.models.save_model(model, str(OUT_ROOT / "saved_models" / "model"))


if __name__ == "__main__":
    main()
