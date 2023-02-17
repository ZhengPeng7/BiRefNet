#!/bin/bash
# Run script
method="$1"
epochs=200
val_last=40
step=20

# Train
python train.py --ckpt_dir ckpt/${method} --epochs ${epochs} --testsets DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4


nvidia-smi
hostname