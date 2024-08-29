#!/bin/sh
# Example: ./sub.sh tmp_proj 0,1,2,3 3 --> Use 0,1,2,3 for training, release GPUs, use GPU:3 for inference.

# module load gcc/11.2.0 cuda/11.8 cudnn/8.6.0_cu11x && cpu_core_num=6
module load compilers/cuda/11.8 compilers/gcc/12.2.0 cudnn/8.4.0.27_cuda11.x && cpu_core_num=32

export PYTHONUNBUFFERED=1

method=${1:-"BSL"}
devices=${2:-0}
gpu_num=$(($(echo ${devices%%,} | grep -o "," | wc -l)+1))

sbatch --nodes=1 -p vip_gpu_ailab -A ai4bio \
    --gres=gpu:${gpu_num} --ntasks-per-node=1 --cpus-per-task=$((gpu_num*cpu_core_num)) \
    ./train_test.sh ${method} ${devices}

hostname
