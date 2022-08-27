import json
from pathlib import Path

import tensorflow as tf

image_extensions = [".jpg"]


def read_image(image_path: tf.Tensor) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    return image


def load_dataset(dataset_root: Path) -> tf.data.Dataset:
    images = []
    jsons = []

    for image in dataset_root.iterdir():
        if image.suffix in image_extensions:
            images.append(image)

            json_file = dataset_root / f"{image.stem}.json"
            if not json_file.exists():
                raise RuntimeError(f"JSON associated with {str(image)} doesnt exist!")

            jsons.append(json.loads(json_file.read_bytes()))

    return tf.data.Dataset.from_tensor_slices(
        (
            {
                "path": [str(image) for image in images],
            },
            {
                "visible": [item["visible"] for item in jsons],
                "bbox": [item["bbox"] for item in jsons],
            },
        )
    )


def prepare_dataset(dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    dataset.shuffle(tf.data.experimental.cardinality(dataset))

    # read images
    dataset = dataset.map(
        lambda x, y: (
            {
                "image": read_image(x["path"]),
            },
            y,
        )
    )

    dataset = dataset.filter(lambda x, y: y["visible"] == 1)

    dataset = dataset.map(
        lambda x, y: (
            {
                "image": tf.image.resize(x["image"], size=(224, 224)),
            },
            {
                "bbox": y["bbox"],
            },
        )
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
