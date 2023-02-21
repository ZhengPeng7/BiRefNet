#!/bin/bash
# Run script
method="$1"
epochs=120
val_last=30
step=10

# Train
python train.py --ckpt_dir ckpt/${method} --epochs ${epochs} --testsets DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4


nvidia-smi
hostname