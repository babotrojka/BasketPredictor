"""
This one takes moves all images from src/{train, val, test}
to dest folder in case that json of src image is not visible
"""
import argparse
import json
import shutil
from pathlib import Path

valid_extensions = [".jpg", ".png"]


def main(
    src_folder: str,
    dest_folder: str,
) -> None:
    src_folder = Path(src_folder)
    dest_folder = Path(dest_folder)

    if not dest_folder.exists():
        dest_folder.mkdir(parents=True)

    count = 0
    for subfolder in ["train", "test", "val"]:
        src = src_folder / subfolder
        for image in src.iterdir():
            if image.suffix not in valid_extensions:
                continue

            json_file = image.parent / f"{image.stem}.json"
            json_read = json.loads(json_file.read_bytes())

            if json_read["visible"] == 0:
                dest_file = dest_folder / f"frame_{str(count).zfill(4)}.jpg"
                while dest_file.exists():
                    count += 1
                    dest_file = dest_folder / f"frame_{str(count).zfill(4)}.jpg"

                shutil.move(image, dest_file)
                count += 1

                json_file.unlink()

    print(f"Transfer finished! Total transfered {count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-s",
        dest="src_folder",
        required=True,
    )
    parser.add_argument(
        "--dest",
        "-d",
        dest="dest_folder",
        required=True,
    )
    main(**vars(parser.parse_args()))
