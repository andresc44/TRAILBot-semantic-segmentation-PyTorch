#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=0 python train.py --model psp \
    --backbone resnet50 --dataset trails \
    --lr 0.004 --epochs 10