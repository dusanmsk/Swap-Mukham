
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

from nsfw_detector import get_nsfw_detector
from face_swapper import Inswapper, paste_to_whole, place_foreground_on_background
from face_analyser import detect_conditions, get_analysed_data, swap_options_list
from face_enhancer import get_available_enhancer_names, load_face_enhancer_model
from face_parsing import init_parser, swap_regions, mask_regions, mask_regions_to_list, SoftErosion
from utils import trim_video, StreamerThread, ProcessBar, open_directory, split_list_by_lengths, merge_img_sequence_from_ref
from utils import measure_time


## ------------------------------ DEFAULTS ------------------------------

USE_COLAB = False
USE_CUDA = False    # TODO arg
BATCH_SIZE = 32 # TODO arg
CPU_COUNT = os.cpu_count()
FACE_ANALYSER_THREADS = 8 # TODO param s defaultom
WORKSPACE = None
OUTPUT_FILE = None
CURRENT_FRAME = None
STREAMER = None
DETECT_CONDITION = "best detection"
DETECT_SIZE = 640
DETECT_THRESH = 0.6
NUM_OF_SRC_SPECIFIC = 10
MASK_INCLUDE = [
    "Skin",
    "R-Eyebrow",
    "L-Eyebrow",
    "L-Eye",
    "R-Eye",
    "Nose",
    "Mouth",
    "L-Lip",
    "U-Lip"
]
MASK_SOFT_KERNEL = 17
MASK_SOFT_ITERATIONS = 7
MASK_BLUR_AMOUNT = 20

FACE_SWAPPER = None
FACE_ANALYSER = None
FACE_ENHANCER = None
FACE_PARSER = None
NSFW_DETECTOR = None
FACE_ENHANCER_LIST = ["NONE"]
FACE_ENHANCER_LIST.extend(get_available_enhancer_names())

## ------------------------------ SET EXECUTION PROVIDER ------------------------------
# Note: Non CUDA users may change settings here

PROVIDER = ["CPUExecutionProvider"]

if USE_CUDA:
    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        print("\n********** Running on CUDA **********\n")
        PROVIDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        USE_CUDA = False
        print("\n********** CUDA unavailable **********\n")
        sys.exit(1)
else:
    USE_CUDA = False
    print("\n********** Running on CPU **********\n")

device = "cuda" if USE_CUDA else "cpu"
EMPTY_CACHE = lambda: torch.cuda.empty_cache() if device == "cuda" else None

## ------------------------------ LOAD MODELS ------------------------------

def load_face_analyser_model(name="buffalo_l"):
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name=name, providers=PROVIDER)
        FACE_ANALYSER.prepare(
            ctx_id=0, det_size=(DETECT_SIZE, DETECT_SIZE), det_thresh=DETECT_THRESH
        )


def load_face_swapper_model(path="./assets/pretrained_models/inswapper_128.onnx"):
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        batch = int(BATCH_SIZE) if device == "cuda" else 1
        FACE_SWAPPER = Inswapper(model_file=path, batch_size=batch, providers=PROVIDER)


def load_face_parser_model(path="./assets/pretrained_models/79999_iter.pth"):
    global FACE_PARSER
    if FACE_PARSER is None:
        FACE_PARSER = init_parser(path, mode=device)

def load_nsfw_detector_model(path="./assets/pretrained_models/nsfwmodel_281.pth"):
    global NSFW_DETECTOR
    if NSFW_DETECTOR is None:
        NSFW_DETECTOR = get_nsfw_detector(model_path=path, device=device)


## ------------------------------ MAIN PROCESS ------------------------------


def process(
    input_type,
    image_path,
    video_path,
    directory_path,
    source_path,
    output_path,
    output_name,
    keep_output_sequence,
    condition,
    age,
    distance,
    face_enhancer_name,
    enable_face_parser,
    mask_includes,
    mask_soft_kernel,
    mask_soft_iterations,
    blur_amount,
    face_scale,
    enable_laplacian_blend,
    crop_top,
    crop_bott,
    crop_left,
    crop_right,
    *specifics,
):
    global WORKSPACE
    global OUTPUT_FILE
    global PREVIEW
    WORKSPACE, OUTPUT_FILE, PREVIEW = None, None, None
    start_time = time.time()
    total_exec_time = lambda start_time: divmod(time.time() - start_time, 60)
    get_finsh_text = lambda start_time: f"✔️ Completed in {int(total_exec_time(start_time)[0])} min {int(total_exec_time(start_time)[1])} sec."

    ## ------------------------------ PREPARE INPUTS & LOAD MODELS ------------------------------

    yield "### \n ⌛ Loading face analyser model..."
    load_face_analyser_model()
    measure_time("Load 2")

    yield "### \n ⌛ Loading face swapper model..."
    load_face_swapper_model()
    measure_time("Load 3")

    if face_enhancer_name != "NONE":
        yield f"### \n ⌛ Loading {face_enhancer_name} model..."
        FACE_ENHANCER = load_face_enhancer_model(name=face_enhancer_name, device=device)
    else:
        FACE_ENHANCER = None

    if enable_face_parser:
        yield "### \n ⌛ Loading face parsing model..."
        load_face_parser_model()

    measure_time("Load 4")

    includes = mask_regions_to_list(mask_includes)
    smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=int(mask_soft_iterations)).to(device) if mask_soft_iterations > 0 else None
    specifics = list(specifics)
    half = len(specifics) // 2
    sources = specifics[:half]
    specifics = specifics[half:]

    measure_time("Measure 1")

    ## ------------------------------ ANALYSE & SWAP FUNC ------------------------------

    def swap_process(image_sequence):

        yield "### \n ⌛ Analysing face data..."
        if condition != "Specific Face":
            source_data = source_path, age
        else:
            source_data = ((sources, specifics), distance)
            
        analysed_targets, analysed_sources, whole_frame_list, num_faces_per_frame = get_analysed_data(
            FACE_ANALYSER,
            image_sequence,
            source_data,
            swap_condition=condition,
            detect_condition=DETECT_CONDITION,
            scale=face_scale
        )
        measure_time("Measure 1a")

        yield "### \n ⌛ Swapping faces..."
        preds, aimgs, matrs = FACE_SWAPPER.batch_forward(whole_frame_list, analysed_targets, analysed_sources)
        EMPTY_CACHE()

        measure_time("Measure 2")

        if enable_face_parser:
            yield "### \n ⌛ Applying face-parsing mask..."
            for idx, (pred, aimg) in tqdm(enumerate(zip(preds, aimgs)), total=len(preds), desc="Face parsing"):
                preds[idx] = swap_regions(pred, aimg, FACE_PARSER, smooth_mask, includes=includes, blur=int(blur_amount))
        EMPTY_CACHE()

        measure_time("Measure 3")

        if face_enhancer_name != "NONE":
            yield f"### \n ⌛ Enhancing faces with {face_enhancer_name}..."
            for idx, pred in tqdm(enumerate(preds), total=len(preds), desc=f"{face_enhancer_name}"):
                enhancer_model, enhancer_model_runner = FACE_ENHANCER
                pred = enhancer_model_runner(pred, enhancer_model)
                preds[idx] = cv2.resize(pred, (512,512))
                aimgs[idx] = cv2.resize(aimgs[idx], (512,512))
                matrs[idx] /= 0.25

        EMPTY_CACHE()

        split_preds = split_list_by_lengths(preds, num_faces_per_frame)
        del preds
        split_aimgs = split_list_by_lengths(aimgs, num_faces_per_frame)
        del aimgs
        split_matrs = split_list_by_lengths(matrs, num_faces_per_frame)
        del matrs

        measure_time("Measure 4")

        yield "### \n ⌛ Post-processing..."
        def post_process(frame_idx, frame_img, split_preds, split_aimgs, split_matrs, enable_laplacian_blend, crop_top, crop_bott, crop_left, crop_right):
            whole_img_path = frame_img
            whole_img = cv2.imread(whole_img_path)
            for p, a, m in zip(split_preds[frame_idx], split_aimgs[frame_idx], split_matrs[frame_idx]):
                whole_img = paste_to_whole(p, a, m, whole_img, laplacian_blend=enable_laplacian_blend, crop_mask=(crop_top, crop_bott, crop_left, crop_right))
            cv2.imwrite(whole_img_path, whole_img)

        def concurrent_post_process(image_sequence, split_preds, split_aimgs, split_matrs, enable_laplacian_blend, crop_top, crop_bott, crop_left, crop_right):
            with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
                futures = []
                for idx, frame_img in enumerate(image_sequence):
                    future = executor.submit(
                        post_process,
                        idx,
                        frame_img,
                        split_preds,
                        split_aimgs,
                        split_matrs,
                        enable_laplacian_blend,
                        crop_top,
                        crop_bott,
                        crop_left,
                        crop_right
                    )
                    futures.append(future)

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Post-Processing"):
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"An error occurred: {e}")

        concurrent_post_process(
            image_sequence,
            split_preds,
            split_aimgs,
            split_matrs,
            enable_laplacian_blend,
            crop_top,
            crop_bott,
            crop_left,
            crop_right
        )
        measure_time("Measure 5")


    ## ------------------------------ IMAGE ------------------------------

    if input_type == "Image":
        target = cv2.imread(image_path)
        output_file = os.path.join(output_path, output_name + ".png")
        cv2.imwrite(output_file, target)

        for info_update in swap_process([output_file]):
            yield info_update

        OUTPUT_FILE = output_file
        WORKSPACE = output_path
        PREVIEW = cv2.imread(output_file)[:, :, ::-1]
        yield
        #yield get_finsh_text(start_time), *ui_after()

    ## ------------------------------ VIDEO ------------------------------

    elif input_type == "Video":
        temp_path = os.path.join(output_path, output_name, "sequence")
        os.makedirs(temp_path, exist_ok=True)

        measure_time("Before extract")
        yield "### \n ⌛ Extracting video frames... on " + str(CPU_COUNT)
        save_executor = ThreadPoolExecutor(max_workers=CPU_COUNT)
        timestart = round(time.time())
        image_sequence = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_idx = 0
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:break
                curr_idx_str=format(curr_idx, "015")
                frame_path = os.path.join(temp_path, f"frame_{curr_idx_str}.jpg")
                save_executor.submit(cv2.imwrite, frame_path, frame)
                image_sequence.append(frame_path)
                curr_idx += 1
                total_frames -= 1
                pbar.update(1)
        cap.release()
        cv2.destroyAllWindows()
        save_executor.shutdown()
        timeend = round(time.time())
        measure_time("After extract")
        print ("duration: " + str((timeend - timestart)))
        

        for info_update in swap_process(image_sequence):
            yield info_update

        yield "### \n ⌛ Merging sequence..."
        output_video_path = os.path.join(output_path, output_name + ".mp4")
        measure_time("Merge 1")
        merge_img_sequence_from_ref(video_path, image_sequence, output_video_path)
        measure_time("Merge2 2")

        if os.path.exists(temp_path) and not keep_output_sequence:
            yield "### \n ⌛ Removing temporary files..."
            shutil.rmtree(temp_path)

        WORKSPACE = output_path
        OUTPUT_FILE = output_video_path

        yield get_finsh_text(start_time)

    ## ------------------------------ DIRECTORY ------------------------------

    elif input_type == "Directory":
        extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "ico", "webp"]
        temp_path = os.path.join(output_path, output_name)
        os.makedirs(temp_path,exist_ok=True)

        file_paths =[]
        for file_path in glob.glob(os.path.join(directory_path, "*")):
            if any(file_path.lower().endswith(ext) for ext in extensions):
                img = cv2.imread(file_path)
                new_file_path = os.path.join(temp_path, os.path.basename(file_path))
                cv2.imwrite(new_file_path, img)
                file_paths.append(new_file_path)

        for info_update in swap_process(file_paths):
            yield info_update

        PREVIEW = cv2.imread(file_paths[-1])[:, :, ::-1]
        WORKSPACE = temp_path
        OUTPUT_FILE = file_paths[-1]

        yield get_finsh_text(start_time)

    ## ------------------------------ STREAM ------------------------------

    elif input_type == "Stream":
        pass



def stop_running():
    global STREAMER
    if hasattr(STREAMER, "stop"):
        STREAMER.stop()
        STREAMER = None
    return "Cancelled"



def runProject(project):
    print("Running project: " + json.dumps(project, indent=4, sort_keys=True))

    for i in process(
        input_type = project['input_type'],
        video_path = project['video_path'],
        image_path = project['image_path'],
        source_path = project['source_path'],
        directory_path = project['directory_path'],
        output_path = project['output_path'],
        output_name = project['output_name'],
        keep_output_sequence = project['keep_output_sequence'],
        mask_includes = project['mask_includes'],
        mask_soft_kernel = project['mask_soft_kernel'],
        mask_soft_iterations = project['mask_soft_iterations'],
        age = project['age'],
        blur_amount = project['blur_amount'],
        condition = project['condition'],
        crop_bott = project['crop_bott'],
        crop_left = project['crop_left'],
        crop_right = project['crop_right'],
        crop_top = project['crop_top'],
        distance = project['distance'],
        enable_face_parser = project['enable_face_parser'],
        enable_laplacian_blend = project['enable_laplacian_blend'],
        face_enhancer_name = project['face_enhancer_name'],
        face_scale = project['face_scale']
    ):
        print("cycle" + str(i))

    print("Process end")

