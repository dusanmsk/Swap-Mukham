import numpy as np
import gradio as gr
import pathlib
import os
import logging as log
import batch
import shutil
import glob
import json

# TODO vypnut public

log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
    handlers=[
        log.FileHandler("app_xxx.log"),
        log.StreamHandler()
    ]
)

image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "ico", "webp"]

# path to folder with destination faces
#FACES_PATH='/home/msk/tmp/gradio/faces'
FACES_PATH = "/home/msk/encfs/projects/pic/faces"

# path to source files
#SOURCES_PATH='/home/msk/tmp/gradio/sources'
SOURCES_PATH = "/home/msk/encfs/projects/pic/src"

TMPDIR="/tmp/faceswap"      # todo ~/encfs
OUTPUT_DIR="/home/msk/encfs/projects/pic/batch/out"
PROJECT_DIR="/home/msk/encfs/projects/pic/batch/"

class c_UI:
    sources_gallery = None
    faces_gallery = None
    image_preview = None
    gfpgan_checkbox = None

class c_State:
    face_images = []
    selected_face = None
    source_images = []
    selected_source = None

UI = c_UI()
STATE = c_State()

def findImages(path):
    global image_extensions
    images = []
    for filename in glob.glob(path+"/**", recursive=True):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            images.append(filename)   
    return images

def processSelectedSourceEvent(evt: gr.SelectData):
    global STATE
    STATE.selected_source = STATE.source_images[evt.index] 
    log.debug(f"Selected {STATE.selected_source} source" )
    return STATE.selected_source

def processSelectedFaceEvent(evt: gr.SelectData):
    global STATE
    STATE.selected_face = STATE.face_images[evt.index] 
    log.debug(f"Selected {STATE.selected_face} face" )
    return STATE.selected_face


def showSelectedSourceImage(imageView):
    global STATE
    global UI
    # todo return gr.update(....)

def prepareProject(gfpgan):
    return {
        "input_type": "",
        "video_path": "",
        "image_path" : None,
        "source_path" : STATE.selected_face,
        "directory_path" : "",
        "output_path" : "",
        "output_name" : "",
        "keep_output_sequence": False,
        "mask_includes" : [],
        "mask_soft_kernel" : 17.0,
        "mask_soft_iterations" : 7,
        "age" : 25,
        "blur_amount" : 20.0,
        "condition" : "All Female",   # todo
        "crop_bott" : 0,
        "crop_left" : 0,
        "crop_right" : 0,
        "crop_top" : 0,
        "distance" : 0.6,
        "enable_face_parser" : False, # todo checkbox
        "enable_laplacian_blend" : True,
        "face_enhancer_name" : "GFPGAN" if gfpgan else "NONE",
        "face_scale" : 1

    }    

def previewTransform(gfpgan):
    log.debug(f"Transforming source {STATE.selected_source} with face {STATE.selected_face}")
    # TODO process
    output_name="preview"
    output_filename=os.path.join(TMPDIR, output_name+".png")
    os.makedirs(TMPDIR, exist_ok=True)
    shutil.copy(STATE.selected_source, output_filename)
    project = prepareProject(gfpgan=gfpgan)
    project["input_type"] = "Image"
    project["image_path"] = STATE.selected_source
    project["output_path"] = TMPDIR
    project["output_name"] = output_filename
    batch.runProject(project=project)
    return gr.update(value=batch.PREVIEW, visible=True)

def submitBatchProject(gfpgan):
    source_dir = os.path.dirname(STATE.selected_source)
    src_name = os.path.basename(source_dir)
    src_hash=str(abs(hash(source_dir)))
    face_name = os.path.basename(STATE.selected_face)
    face_name = face_name[0:face_name.rindex('.')]
    output_dir = os.path.join(OUTPUT_DIR, face_name)
    project = prepareProject(gfpgan=gfpgan)
    project["input_type"] = "Directory"
    project["directory_path"] = source_dir
    project["source_path"] = STATE.selected_face
    project["output_path"] = output_dir
    #os.makedirs(output_dir, exist_ok=True)
    project_file_name=os.path.join(PROJECT_DIR, f"{face_name}-{src_name}.json")
    with open(project_file_name, "w") as outfile:
        outfile.write(json.dumps(project, indent=4))

def reload():
    global STATE
    STATE.face_images = findImages(FACES_PATH)
    STATE.source_images = findImages(SOURCES_PATH)
    return [STATE.source_images, STATE.face_images]    


def setupUi():
    global UI
    global STATE

    css = '''
    <!-- this makes the items inside the gallery to shrink -->
    div#sources_gallery div.grid {
        height: 64px;
        width: 180px;
    }

    <!-- this makes the gallery's height to shrink -->
    div#sources_gallery > div:nth-child(3) {
        min-height: 172px !important;
    }

    <!-- this makes the gallery's height to shrink when you click one image to view it bigger -->
    div#sources_gallery > div:nth-child(4) {
        min-height: 172px !important;
    }
    '''

    # TODO
    css = ''

    with gr.Blocks(css=css, layout="vertical") as interface:
        with gr.Row():
            with gr.Column():
                UI.sources_gallery = gr.Gallery(allow_preview=False, elem_id='sources_gallery').style(columns=[4], rows=[2], object_fit="contain", height="auto")

            with gr.Column():
                with gr.Row():            
                    with gr.Column():
                        reload_button = gr.Button("Reload").style(full_width=False)
                    with gr.Column():
                        submit_project_button = gr.Button("Submit").style(full_width=False)

                with gr.Row():            
                    UI.gfpgan_checkbox = gr.Checkbox(
                        label="GFPGAN", value=False, interactive=True
                    )
                with gr.Row():            
                    UI.image_preview=gr.Image(tool=["select"], interactive=True).style(height=500,width=800)
                with gr.Row():            
                    preview_button = gr.Button("Preview").style(full_width=False)
        with gr.Row():
            UI.faces_gallery = gr.Gallery(allow_preview=False).style(columns=[10], rows=[2], object_fit="contain",
                                                                     height="auto")

                    
                
        UI.sources_gallery.select(processSelectedSourceEvent, None, None)
        UI.faces_gallery.select(processSelectedFaceEvent, None, None)
        preview_button.click(previewTransform, [UI.gfpgan_checkbox], UI.image_preview)
        submit_project_button.click(submitBatchProject, [UI.gfpgan_checkbox], None)
        reload_button.click(reload, None, [UI.sources_gallery, UI.faces_gallery])
        # populate galleries
        interface.load(lambda: STATE.source_images, None, UI.sources_gallery)
        interface.load(lambda: STATE.face_images, None, UI.faces_gallery)

    interface.launch(share=False, width="100%", height=200)


def main():
    STATE.face_images = findImages(FACES_PATH)
    STATE.source_images = findImages(SOURCES_PATH)
    setupUi()
    

main()    