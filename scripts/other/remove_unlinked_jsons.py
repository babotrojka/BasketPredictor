"""
Removes jsons with no image connected to them
"""
import argparse
from pathlib import Path

valid_extensions = [".json"]


def main(
    src_folder: str,
) -> None:
    src_folder = Path(src_folder)

    count = 0
    for subfolder in ["train", "test", "val"]:
        src = src_folder / subfolder
        for json_file in src.iterdir():
            if json_file.suffix not in valid_extensions:
                continue

            image = json_file.parent / f"{json_file.stem}.jpg"
            if not image.exists():
                json_file.unlink()
                count += 1

    print(f"Unlinking finished! Total unlinked {count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-s",
        dest="src_folder",
        required=True,
    )
    main(**vars(parser.parse_args()))
