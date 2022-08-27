import argparse
import shutil
from pathlib import Path

import numpy as np

valid_extensions = [".jpg", ".png"]


def extract_images_and_json(folder: Path) -> np.ndarray:
    x_all = []
    for snimak in folder.iterdir():
        for img in snimak.iterdir():
            if img.suffix in valid_extensions:
                json_file = Path(
                    snimak,
                    f"{img.stem}.json",
                )
                x_all.append((img, json_file))

    return np.array(x_all)


def copy_to_dest(files: np.ndarray, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for img, json in files:
        shutil.copy(img, dest)
        shutil.copy(json, dest)


def split(args) -> None:
    splits = {"train": args.train}

    if args.val:
        splits["val"] = args.val
    if args.test:
        splits["test"] = args.test

    splits_sum = sum(splits.values())
    if splits_sum != 1:
        raise AttributeError("Sum of splits must be equal to 1")

    folder = Path(args.folder)
    dest_root = Path(args.dest)

    x_all = extract_images_and_json(folder)

    trains = np.random.choice(
        len(x_all), size=int(splits["train"] * len(x_all)), replace=False
    )

    rest = np.setdiff1d(np.arange(len(x_all)), trains)
    rest_vals = splits["val"] / (splits["val"] + splits["test"])
    vals = np.random.choice(rest, size=int(rest_vals * len(rest)), replace=False)

    tests = np.setdiff1d(rest, vals)

    train = x_all[trains]
    val = x_all[vals]
    test = x_all[tests]

    # print(Path(dest_root, 'train'))
    copy_to_dest(train, dest_root / "train")
    copy_to_dest(val, dest_root / "val")
    copy_to_dest(test, dest_root / "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", required=True)
    parser.add_argument("--dest", "-d", required=True)
    parser.add_argument("--train", type=float, required=True)
    parser.add_argument("--val", type=float, required=False)
    parser.add_argument("--test", type=float, required=False)
    split(parser.parse_args())
