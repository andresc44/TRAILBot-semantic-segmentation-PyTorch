#!/usr/bin/env bash

# eval
CUDA_VISIBLE_DEVICES=0 python eval.py --model new_deeplabv3 \
    --backbone resnet50 --dataset trails \
    --lr 0.004 --epochs 100