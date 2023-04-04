#!/bin/bash
# Run script
method="$1"
epochs=100
val_last=20
step=10

# Train
CUDA_VISIBLE_DEVICES=$2 python train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
    --testsets DIS-VD+DIS-TE1

echo Finished at $(date)
