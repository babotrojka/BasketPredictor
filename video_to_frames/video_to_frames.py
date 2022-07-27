import posix

import cv2
import os
import sys
import shutil

FOLDER_PATH = "../../snimci/"

"""
    Given folder with videos in it, creates a 
    folder_name_frames folder in it which consists
    of folders for each video. Each folder then has all the
    frames of corresponding video
"""

def video_to_frame(video: posix.DirEntry, path_to_save: str, extension: str = ".mp4") -> None:
    video_name = video.name[:video.name.index(extension)]
    video_frames_folder = f"{path_to_save}/{video_name}"

    if os.path.exists(video_frames_folder):
        shutil.rmtree(video_frames_folder)
    os.mkdir(video_frames_folder)

    vidcap = cv2.VideoCapture(video.path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{video_frames_folder}/frame_{str(count).zfill(4)}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

def folder_to_frames(path_folder: str, path_frames_folder: str = None, extension: str = '.mp4') -> None:
    if path_frames_folder is None:
        path_frames_folder = path_folder

    folder_name = os.path.basename(path_frames_folder) if path_frames_folder[-1] != '/' else os.path.basename(path_frames_folder[:-1])
    frames_folder = f"{path_frames_folder}/{folder_name}_frames"
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)

    os.mkdir(frames_folder)
    for video in os.scandir(path_folder):
        if not video.name.endswith(extension):
            continue

        video_to_frame(video, frames_folder)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        folder = FOLDER_PATH
    else:
        folder = sys.argv[0]

    folder_to_frames(folder)

    print('Writing finished!')
    
