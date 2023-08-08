import cv2
import glob
import os
import shutil

# detekuje tvar vo videu a zapise video len s tvarou
# len scratch, bude sluzit na detekciu blokov videa kde je tvar

face_cascade = cv2.CascadeClassifier('scratch/haarcascade_frontalface_default.xml')
dir="/home/msk/work/misc/faceswap/app/Result/sequence"
out="/home/msk/work/misc/faceswap/app/Result/sequence/detected/"

def detectFace(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if faces == ():
        return False
    return True    


def detectInVideo():

    video = cv2.VideoCapture("/home/msk/tmp/facetest.mp4")
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)
    result = cv2.VideoWriter('filename.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)


    frames = []
    c = 1
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            break
        if c%fps*2==0:
            if detectFace(frame):
                frames.append(frame)
                result.write(frame)
        c+=1


def detectInDirectory():
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(f"{dir}/**.jpg"):
        frame=cv2.imread(f)
        if detectFace(frame):
            shutil.move(f,out)

detectInDirectory()
