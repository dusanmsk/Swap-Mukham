# initialize GPU environment

apt-get update
apt-get -y install build-essential python3-opencv megatools byobu
apt-get -y clean

conda create -n swapp-gpu python=3.10 -y
conda activate swapp-gpu
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

