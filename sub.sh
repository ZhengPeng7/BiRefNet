#!/bin/sh
# Example: ./sub.sh tmp_proj 0,1,2,3 3 --> Use 0,1,2,3 for training, release GPUs, use GPU:3 for inference.

module load compilers/cuda/11.8

export PYTHONUNBUFFERED=1
export LD_PRELOAD=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64/libstdc++.so.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/miniconda3/lib:/home/bingxing2/apps/cudnn/8.4.0.27_cuda11.x/lib

method=${1:-"BSL"}
devices=${2:-0}

sbatch --nodes=1 -p vip_gpu_ailab -A ai4bio \
--ntasks-per-node=1 \
--gres=gpu:$(($(echo ${devices%%,} | grep -o "," | wc -l)+1)) \
--cpus-per-task=32 \
./train_test.sh ${method} ${devices}

hostname
