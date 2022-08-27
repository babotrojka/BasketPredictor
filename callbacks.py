import shutil
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import patches
from matplotlib.figure import Figure


def figure_to_numpy(fig: Figure) -> np.ndarray:
    fig.canvas.draw()

    np_canvas: np.ndarray = np.frombuffer(
        fig.canvas.tostring_rgb(),
        dtype=np.uint8,
    )
    np_image = np_canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return np_image


def create_patch_from_bbox(
    image_shape: tuple, bbox: tf.Tensor, color: Union[str, tuple], linewidth: int = 1
) -> patches.Rectangle:
    image_width, image_height, _ = image_shape
    y = bbox[0] * image_width
    x = bbox[1] * image_height
    h = bbox[2] * image_width - y
    w = bbox[3] * image_height - x

    return patches.Rectangle(
        (x, y), w, h, edgecolor=color, linewidth=linewidth, facecolor="none"
    )


def create_fig_with_bbox_patches(
    image: np.ndarray, bbox_true: np.ndarray, bbox_pred: np.ndarray
) -> plt.Figure:
    fig, axis = plt.subplots()
    axis.imshow(image)

    patch_true = create_patch_from_bbox(image.shape, bbox_true, color="b")
    axis.add_patch(patch_true)

    patch_pred = create_patch_from_bbox(image.shape, bbox_pred, color="r")
    axis.add_patch(patch_pred)

    return fig


class ReportImages(tf.keras.callbacks.Callback):
    def __init__(self, root: Path, data: tf.data.Dataset):
        self.folder = root / "images"
        if self.folder.exists():
            shutil.rmtree(self.folder)
        self.folder.mkdir(parents=True, exist_ok=False)

        self.data = data

    def on_epoch_end(self, epoch: int, _: Any):
        out_folder = self.folder / f"epoch_{epoch + 1}"
        out_folder.mkdir(exist_ok=True)

        for i, (x_b, y_b) in enumerate(self.data):
            y_hat = self.model.predict_on_batch(x_b)
            for j in range(x_b["image"].shape[0]):
                fig = create_fig_with_bbox_patches(
                    image=x_b["image"][j].numpy().astype(np.uint16),
                    bbox_true=y_b["bbox"][j],
                    bbox_pred=y_hat[j],
                )

                plt.savefig(out_folder / f"{i:2d}_{j:3d}.jpg")
                plt.close(fig)


class TensorboardReportImages(tf.keras.callbacks.Callback):
    def __init__(self, logdir: Path, data: tf.data.Dataset) -> None:
        self.file_writer = tf.summary.create_file_writer(str(logdir))
        self.data = data

    def on_epoch_end(self, epoch: int, _: Any) -> None:
        for x_b, y_b in self.data.as_numpy_iterator():
            y_hat = self.model.predict_on_batch(x_b)

            image_shape = x_b["image"].shape
            batch_len = image_shape[0]
            images = []
            for j in range(batch_len):
                fig = create_fig_with_bbox_patches(
                    image=x_b["image"][j].astype(np.uint16),
                    bbox_true=y_b["bbox"][j],
                    bbox_pred=y_hat[j],
                )

                numpy_img = figure_to_numpy(fig)

                images.append(numpy_img)

                plt.close(fig)

            with self.file_writer.as_default():
                tf.summary.image(
                    "Training data",
                    np.array(images),
                    max_outputs=batch_len,
                    step=epoch,
                )
