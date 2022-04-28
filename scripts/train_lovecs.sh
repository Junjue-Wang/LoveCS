#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='st.lovecs.2CZ.lovecs'
python LoveCS_train.py --config_path=${config_path}
