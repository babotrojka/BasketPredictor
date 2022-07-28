import os
import argparse
import json
import numpy as np


def read_extract_write(source_file: str, dest_file: str, class_to_extract):
    with open(source_file, "r") as f:
        json_content = json.load(f)

    classes = json_content["classes"][0]
    if class_to_extract in classes:
        index = classes.index(class_to_extract)
        result = {
            "visibile": 1,
            "bbox": json_content["boxes"][0][index]
        }
    else:
        result = {
            "visible": 0,
            "bbox": np.zeros(4).tolist()
        }

    with open(dest_file, "w") as fw:
        fw.write(json.dumps(result))

def main(args):
    json_folder = args.json
    images_folder = args.images
    class_to_extract = args.e_class
    if not os.path.exists(json_folder) or not os.path.exists(images_folder):
        print("Json folder or images folder does not exist. Please provide valid folder")
        exit(0)

    folder_num = len(os.listdir(json_folder))
    for i, video_folder in enumerate(os.scandir(json_folder)):
        dest_folder = os.path.join(images_folder, video_folder.name)
        if not os.path.exists(dest_folder):
            print("Structure of image folder is not the same as of json folder")
            exit(0)

        for json_txt_file in os.scandir(video_folder):
            dest_file = os.path.join(dest_folder, f"{os.path.splitext(json_txt_file.name)[0]}.json")
            read_extract_write(json_txt_file.path, dest_file, class_to_extract)

        print(f"Finished with folder {video_folder.name} {i+1}/{folder_num}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", "-j", help="Json parent folder", required=True)
    parser.add_argument("--images", "-i", help="Images parent folder. Structure must be the same as json folder", required=True)
    parser.add_argument("--e_class", "-c", help="Class you want to extract", type=int, default=32)
    main(parser.parse_args())
