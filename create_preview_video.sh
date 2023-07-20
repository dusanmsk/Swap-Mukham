#!/bin/bash

# usage:
# create_preview_video.sh /path/to/source_video.mp4 /path/to/dest_video.mp4 [start second, length in seconds, framerate]
# defaults are 10 and 3


INPUT_FILE="$1"
OUTPUT_FILE="$2"
START_SEC="${3:-0}"
PREVIEW_LENGTH="${4:-10}"
FPS="${5:-3}"

duration_sec=`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $INPUT_FILE`
if [ "$START_SEC" == "0" ]; then
    START_SEC=`echo "$duration_sec / 2" | bc`
fi    

ffmpeg_cmd="ffmpeg -i $INPUT_FILE -ss $START_SEC -t $PREVIEW_LENGTH -filter:v fps=$FPS $OUTPUT_FILE"
echo $ffmpeg_cmd
$ffmpeg_cmd

echo
echo "File duration is $duration_sec, extracted from $START_SEC with length $PREVIEW_LENGTH at fps $FPS"



