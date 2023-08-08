
from concurrent.futures import ThreadPoolExecutor
import time
import os
import cv2
import glob
import time
import torch
import shutil
import argparse
import platform
import datetime
import subprocess
import insightface
import onnxruntime
import numpy as np
import gradio as gr
from tqdm import tqdm
import concurrent.futures
from moviepy.editor import VideoFileClip
import json
import logging
import sys
import batch

from nsfw_detector import get_nsfw_detector
from face_swapper import Inswapper, paste_to_whole, place_foreground_on_background
from face_analyser import detect_conditions, get_analysed_data, swap_options_list
from face_enhancer import get_available_enhancer_names, load_face_enhancer_model
from face_parsing import init_parser, swap_regions, mask_regions, mask_regions_to_list, SoftErosion
from utils import trim_video, StreamerThread, ProcessBar, open_directory, split_list_by_lengths, merge_img_sequence_from_ref
from utils import measure_time

## ------------------------------ USER ARGS ------------------------------

parser = argparse.ArgumentParser(description="Swap-Mukham Face Swapper")
parser.add_argument("--batch_size", help="Gpu batch size", default=32)
parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=False)
parser.add_argument("--quick", action="store_true", help="Disable postprocessing", default=False)
parser.add_argument("--project_file", help="Project file to run in headless batch mode")
parser.add_argument("--cpu_count", help="Number of CPUs to run video extraction on", default=1)

user_args = parser.parse_args()
print(user_args)

## ------------------------------ DEFAULTS ------------------------------

global USE_CUDA
global BATCH_SIZE
global CPU_COUNT

PROJECT_FILE = user_args.project_file
QUICK = user_args.quick
USE_CUDA = user_args.cuda
BATCH_SIZE = user_args.batch_size
CPU_COUNT = os.cpu_count()
FACE_ANALYSER_THREADS = 8 # TODO param s defaultom


# TODO !!!!!!!!!!!!!! remove !!!!!!!!!!!!!!
#PROJECT_FILE = "/home/msk/tmp/faceswap/project1/project.json"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# !!!!!!!!!!!!!! END remove !!!!!!!!!!!!!!

if not os.path.isfile(PROJECT_FILE):
    print("Error. Project file doesnt exists.")
    exit(1)
PROJECT_DIR = os.path.dirname(PROJECT_FILE)

def fixProjectPaths(p):
    for i in ['video_path', 'source_path', 'output_path', 'directory_path']:
        if not os.path.isabs(p[i]):
            p[i] = os.path.join(PROJECT_DIR, p[i])
    return p



if __name__ == "__main__":
    print("Running project ${PROJECT_FILE} in batch mode")

    # read project file
    with open(PROJECT_FILE, 'r') as openfile:
       project = json.load(openfile)

    project = fixProjectPaths(project)
    if(QUICK):
        project["face_enhancer_name"] = "NONE"
    print("Running project: " + json.dumps(project, indent=4, sort_keys=True))

    batch.runProject(project)

    print("Process end")

