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

from nsfw_detector import get_nsfw_detector
from face_swapper import Inswapper, paste_to_whole, place_foreground_on_background
from face_analyser import detect_conditions, get_analysed_data, swap_options_list
from face_enhancer import get_available_enhancer_names, load_face_enhancer_model
from face_parsing import init_parser, swap_regions, mask_regions, mask_regions_to_list, SoftErosion
from utils import trim_video, StreamerThread, ProcessBar, open_directory, split_list_by_lengths, merge_img_sequence_from_ref

## ------------------------------ USER ARGS ------------------------------

parser = argparse.ArgumentParser(description="Swap-Mukham Face Swapper")
parser.add_argument("--out_dir", help="Default Output directory", default=os.getcwd())
parser.add_argument("--batch_size", help="Gpu batch size", default=32)
parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=False)
parser.add_argument("--batch", action="store_true", help="Run in headless batch mode", default=False)
parser.add_argument(
    "--colab", action="store_true", help="Enable colab mode", default=False
)
user_args = parser.parse_args()

## ------------------------------ DEFAULTS ------------------------------

USE_COLAB = user_args.colab
USE_CUDA = user_args.cuda
DEF_OUTPUT_PATH = user_args.out_dir
BATCH_SIZE = user_args.batch_size
BATCH_MODE = user_args.batch
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
        print("\n********** CUDA unavailable running on CPU **********\n")
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


load_face_analyser_model()
load_face_swapper_model()

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
    print("process start 2")
    input_type = "Video"
    video_path = '/home/msk/encfs/faceswap/projects/test1/src-short.webm'
    image_path = ''
    source_path = '/home/msk/encfs/faceswap/projects/test1/dst.jpg'
    directory_path = ''
    output_path = '/home/msk/encfs/out'
    output_name = 'Result'
    keep_output_sequence = False
    mask_includes = ['Skin', 'R-Eyebrow', 'L-Eyebrow', 'L-Eye', 'R-Eye', 'Nose', 'Mouth', 'L-Lip', 'U-Lip']
    mask_soft_kernel = 17.0
    mask_soft_iterations = 7.0
    age = 25.0
    blur_amount = 20.0
    condition = 'All Face'
    crop_bott = 0.0
    crop_left = 0.0
    crop_right = 0.0
    crop_top = 0.0
    distance = 0.6
    enable_face_parser = False
    enable_laplacian_blend = True
    #face_enhancer_name = 'GFPGAN'
    face_enhancer_name = "NONE"
    face_scale = 1

    start_time = time.time()
    total_exec_time = lambda start_time: divmod(time.time() - start_time, 60)
    get_finsh_text = lambda start_time: f"✔️ Completed in {int(total_exec_time(start_time)[0])} min {int(total_exec_time(start_time)[1])} sec."

    ## ------------------------------ PREPARE INPUTS & LOAD MODELS ------------------------------
    yield "### \n ⌛ Loading NSFW detector model..."
    load_nsfw_detector_model()

    yield "### \n ⌛ Loading face analyser model..."
    load_face_analyser_model()

    yield "### \n ⌛ Loading face swapper model..."
    load_face_swapper_model()

    if face_enhancer_name != "NONE":
        yield f"### \n ⌛ Loading {face_enhancer_name} model..."
        FACE_ENHANCER = load_face_enhancer_model(name=face_enhancer_name, device=device)
    else:
        FACE_ENHANCER = None

    if enable_face_parser:
        yield "### \n ⌛ Loading face parsing model..."
        load_face_parser_model()

    includes = mask_regions_to_list(mask_includes)
    smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=int(mask_soft_iterations)).to(device) if mask_soft_iterations > 0 else None
    specifics = list(specifics)
    half = len(specifics) // 2
    sources = specifics[:half]
    specifics = specifics[half:]

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

        yield "### \n ⌛ Swapping faces..."
        preds, aimgs, matrs = FACE_SWAPPER.batch_forward(whole_frame_list, analysed_targets, analysed_sources)
        EMPTY_CACHE()

        if enable_face_parser:
            yield "### \n ⌛ Applying face-parsing mask..."
            for idx, (pred, aimg) in tqdm(enumerate(zip(preds, aimgs)), total=len(preds), desc="Face parsing"):
                preds[idx] = swap_regions(pred, aimg, FACE_PARSER, smooth_mask, includes=includes, blur=int(blur_amount))
        EMPTY_CACHE()

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

        yield "### \n ⌛ Post-processing..."
        def post_process(frame_idx, frame_img, split_preds, split_aimgs, split_matrs, enable_laplacian_blend, crop_top, crop_bott, crop_left, crop_right):
            whole_img_path = frame_img
            whole_img = cv2.imread(whole_img_path)
            for p, a, m in zip(split_preds[frame_idx], split_aimgs[frame_idx], split_matrs[frame_idx]):
                whole_img = paste_to_whole(p, a, m, whole_img, laplacian_blend=enable_laplacian_blend, crop_mask=(crop_top, crop_bott, crop_left, crop_right))
            cv2.imwrite(whole_img_path, whole_img)

        def concurrent_post_process(image_sequence, split_preds, split_aimgs, split_matrs, enable_laplacian_blend, crop_top, crop_bott, crop_left, crop_right):
            with concurrent.futures.ThreadPoolExecutor() as executor:
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

        yield get_finsh_text(start_time), *ui_after()

    ## ------------------------------ VIDEO ------------------------------

    elif input_type == "Video":
        temp_path = os.path.join(output_path, output_name, "sequence")
        os.makedirs(temp_path, exist_ok=True)

        yield "### \n ⌛ Extracting video frames..."
        image_sequence = []
        cap = cv2.VideoCapture(video_path)
        curr_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:break
            frame_path = os.path.join(temp_path, f"frame_{curr_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            image_sequence.append(frame_path)
            curr_idx += 1
        cap.release()
        cv2.destroyAllWindows()

        for info_update in swap_process(image_sequence):
            yield info_update

        yield "### \n ⌛ Merging sequence..."
        output_video_path = os.path.join(output_path, output_name + ".mp4")
        merge_img_sequence_from_ref(video_path, image_sequence, output_video_path)

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
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)

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

with gr.Blocks() as demo:
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=process, inputs=None, outputs=None, api_name="greet")


if __name__ == "__main__":
    #if BATCH_MODE:
    print("Running batch mode")
    for i in process(
        input_type = "Video",
        video_path = '/home/msk/encfs/faceswap/projects/test1/src-short.webm',
        image_path = '',
        source_path = '/home/msk/encfs/faceswap/projects/test1/dst.jpg',
        directory_path = '',
        output_path = '/home/msk/encfs/out',
        output_name = 'Result',
        keep_output_sequence = False,
        mask_includes = ['Skin', 'R-Eyebrow', 'L-Eyebrow', 'L-Eye', 'R-Eye', 'Nose', 'Mouth', 'L-Lip', 'U-Lip'],
        mask_soft_kernel = 17.0,
        mask_soft_iterations = 7.0,
        age = 25.0,
        blur_amount = 20.0,
        condition = 'All Face',
        crop_bott = 0.0,
        crop_left = 0.0,
        crop_right = 0.0,
        crop_top = 0.0,
        distance = 0.6,
        enable_face_parser = False,
        enable_laplacian_blend = True,
        #face_enhancer_name = 'GFPGAN',
        face_enhancer_name = "NONE",
        face_scale = 1
    ):
        print("cycle" + str(i))
        
    print("Process end")


if __name__ == "__mainA__":
    demo.launch()