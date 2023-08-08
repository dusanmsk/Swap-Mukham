#!/bin/bash -i
conda activate swap
python batch_app.py --cuda --project $1
