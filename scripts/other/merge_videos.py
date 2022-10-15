import argparse
from pathlib import Path

from moviepy.editor import *


def check_existence(file: str) -> Path:
    file = Path(file)
    if not file.exists():
        raise ValueError(f"Given path {file} does not exist! Exiting...")
    else:
        return file


def main(input_folder: str) -> None:
    input_folder = check_existence(input_folder)

    video_clips = [VideoFileClip(str(video)) for video in input_folder.iterdir()]
    merged = concatenate_videoclips(video_clips)
    merged.write_videofile(str(input_folder / "merged.mp4"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        "-i",
        required=True,
        dest="input_folder",
    )
    main(**vars(parser.parse_args()))
