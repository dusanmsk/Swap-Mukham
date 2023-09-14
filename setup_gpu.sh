#!/bin/bash
set -ie


apt-get update && apt-get -y install build-essential python3-opencv  software-properties-common && apt-add-repository contrib && apt-add-repository non-free && apt-get update
apt-get -y install megatools byobu rclone nano vim mc nvtop yt-dlp
apt-get -y clean

conda create -n swap python=3.10 -y
conda activate swap
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

