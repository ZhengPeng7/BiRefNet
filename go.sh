#!/bin/bash
# Run script
method="$1"
epochs=150
val_last=25

# Train
CUDA_VISIBLE_DEVICES=$2 python train.py --ckpt_dir ckpt/${method} --epochs ${epochs} --testsets DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4

# step=5
# for ((ep=${epochs};ep>${epochs}-${val_last};ep-=${step}))
# do
# pred_dir=/root/autodl-tmp/datasets/dis/preds/${method}/ep${ep}
# # [ ${ep} -gt $[${epochs}-${val_last}] ] && CUDA_VISIBLE_DEVICES=$2 python evaluation/main.py --model_dir ${method}/ep$[${ep}-${step}] --txt_name ${method}; \
# CUDA_VISIBLE_DEVICES=$2 python test.py --pred_dir ${pred_dir} --ckpt ckpt/${method}/ep${ep}.pth
# done

# # python evaluation/main.py --model_dir ${method}/ep$[${ep}-${step}] --txt_name ${method}
# python evaluation/main.py --model_dir ${method} --txt_name ${method}

nvidia-smi
hostname