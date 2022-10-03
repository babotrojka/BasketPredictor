"""
This one will move all images from given folder / {train, val, test}
to dest folder
"""
import argparse
import shutil
import warnings
from pathlib import Path

valid_extensions = [".jpg", ".png"]


def move(src_folder: Path, dest_folder: Path) -> None:
    count = 0
    for image in src_folder.iterdir():
        if image.suffix in valid_extensions:
            json_file = src_folder / f"{image.stem}.json"
            if not json_file.exists():
                warnings.warn(
                    f"Image {image} does not have corresponding json! Skipping..."
                )
                continue

            dest_image = dest_folder / f"frame_{str(count).zfill(4)}.jpg"
            while dest_image.exists():
                count += 1
                dest_image = dest_folder / f"frame_{str(count).zfill(4)}.jpg"

            shutil.move(image, dest_image)

            dest_json = dest_folder / f"frame_{str(count).zfill(4)}.json"
            shutil.move(json_file, dest_json)


def main(
    src_folder: str,
    dest_folder: str,
) -> None:
    src_folder = Path(src_folder)
    dest_folder = Path(dest_folder)

    move(src_folder / "train", dest_folder)
    move(src_folder / "val", dest_folder)
    move(src_folder / "test", dest_folder)

    print("Moving finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        "-s",
        dest="src_folder",
        required=True,
    )
    parser.add_argument("--dest", "-d", dest="dest_folder", required=True)
    main(**vars(parser.parse_args()))
