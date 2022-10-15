import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from callbacks import create_patch_from_bbox, figure_to_numpy

INPUT_WIDTH, INPUT_HEIGHT = 224, 224
video_extensions = [".mp4"]


def check_existence(file: str) -> Path:
    file = Path(file)
    if not file.exists():
        raise ValueError(f"Given path {file} does not exist! Exiting...")
    else:
        return file


def video_to_frames(
    video: Path,
) -> np.ndarray:
    vidcap = cv2.VideoCapture(str(video))
    success, image = vidcap.read()
    frames = []
    while success:
        frames.append(cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT)))
        success, image = vidcap.read()

    vidcap.release()

    return np.array(frames)


def draw_bbox_on_image(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    fig, axis = plt.subplots()
    axis.imshow(image)

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    patch_true = create_patch_from_bbox(image.shape, bbox, color="b")
    axis.add_patch(patch_true)

    image = figure_to_numpy(fig)
    plt.close(fig)
    return image


def process_video(model: tf.keras.Model, video_in: Path, video_out: Path) -> None:
    frames = video_to_frames(video_in)

    y_preds = model.predict(
        frames,
        batch_size=64,
    )

    video = cv2.VideoWriter(
        str(video_out / f"{video_in.stem}_pred.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (640, 480),
    )
    for frame, bbox in zip(frames, y_preds):
        patched_image = draw_bbox_on_image(frame, bbox)
        video.write(patched_image)

    video.release()


def main(
    model_src: str,
    video_input: str,
    video_out: str,
) -> None:
    model_src = check_existence(model_src)
    model = tf.keras.models.load_model(model_src, compile=False)

    video_input = check_existence(video_input)
    video_out = check_existence(video_out)
    print("Model loaded, processing started!")

    if video_out.is_dir():
        for video in video_input.iterdir():
            if video.suffix not in video_extensions:
                continue
            process_video(model, video, video_out)
            print(f"Finished with {video.name}")
    else:
        process_video(model, video_input, video_out)

    print("Processing finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        dest="model_src",
        required=True,
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="video_input",
        required=True,
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="video_out",
        required=True,
    )
    main(**vars(parser.parse_args()))
