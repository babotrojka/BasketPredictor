import datetime
from pathlib import Path

import keras_tuner
import tensorflow as tf
import autokeras as ak
from callbacks import TensorboardReportImages
from dataset import load_dataset, prepare_dataset
from metrics import IOU
from train import make_dataset

DATASET_ROOT = Path("../dataset/")

OUT_ROOT = Path("../out/")


def build_model() -> ak.AutoModel:
    return ak.ImageRegressor(
        output_dim=4,
        metrics=[
            IOU(),
        ],
        objective=keras_tuner.Objective("val_iou", direction="max"),
    )


def main():
    dataset_config = {
        "batch_size": 32,
    }

    dataset = make_dataset(DATASET_ROOT, dataset_config)

    model = build_model()

    model.fit(
        dataset["train"],
        validation_data=dataset["val"],
        # callbacks=[
        #     tf.keras.callbacks.TensorBoard(
        #         TENSORBOARD_LOG,
        #     ),
        #     TensorboardReportImages(
        #         TENSORBOARD_LOG,
        #         dataset["val"].take(1),
        #     ),
        # ],
    )

    model.evaluate(
        dataset["test"],
        batch_size=64,
    )


if __name__ == "__main__":
    main()
