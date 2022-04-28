#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='st.lovecs.2CZ.lovecs'
ckpt_path='./log/sfpn.pth'
python LoveCS_eval.py --config_path=${config_path} \
                       --ckpt_path=${ckpt_path}

