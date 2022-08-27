"""
    Given folder with videos in it, creates a
    folder_name_frames folder in it which consists
    of folders for each video. Each folder then has all the
    frames of corresponding video
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2


def video_to_frame(video: Path, video_frames_folder: Path) -> None:
    if video_frames_folder.exists():
        shutil.rmtree(video_frames_folder)
    video_frames_folder.mkdir()

    vidcap = cv2.VideoCapture(str(video))
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(
            f"{video_frames_folder}/frame_{str(count).zfill(4)}.jpg", image
        )  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


def folder_to_frames(args) -> None:

    path_video = Path(args.video)

    frames_folder = path_video.parent / f"{path_video.stem}_frames"
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)

    os.mkdir(frames_folder)

    video_to_frame(path_video, frames_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", required=True, help="Video to make frames")

    folder_to_frames(parser.parse_args())

    print("Writing finished!")
