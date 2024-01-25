#!/usr/bin/env bash

# eval
CUDA_VISIBLE_DEVICES=0 python3 eval.py --model lednet \
    --backbone resnet50 --dataset trails \
    --lr 0.004 --epochs 30 --batch-size 4