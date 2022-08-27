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


def make_dataset_split(split_root: Path) -> tf.data.Dataset:
    dataset = load_dataset(split_root)
    dataset = prepare_dataset(dataset, batch_size=8)
    return dataset


def make_dataset(dataset_root: Path) -> dict[str, tf.data.Dataset]:
    return {
        split: make_dataset_split(dataset_root / split)
        for split in ["train", "val", "test"]
    }


def main():
    dataset = make_dataset(DATASET_ROOT)

    model = build_model()
    print(model.summary())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3,
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
        epochs=50,
        validation_data=dataset["val"],
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                TENSORBOARD_LOG,
            ),
            TensorboardReportImages(
                TENSORBOARD_LOG,
                dataset["val"],
            ),
        ],
    )


if __name__ == "__main__":
    main()
