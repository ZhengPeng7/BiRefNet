#!/bin/bash

# Run script
# method="$1"

# Train
CUDA_VISIBLE_DEVICES=$1 python train_valid_inference_main.py

