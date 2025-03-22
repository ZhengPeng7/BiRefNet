#!/bin/bash
# Run script
# Settings of training & test for different tasks.
method="$1"
task=$(python3 config.py --print_task)
case "${task}" in
    'DIS5K') epochs=500 && val_last=50 && step=5 ;;
    'COD') epochs=150 && val_last=50 && step=5 ;;
    'HRSOD') epochs=150 && val_last=50 && step=5 ;;
    'General') epochs=200 && val_last=50 && step=5 ;;
    'General-2K') epochs=250 && val_last=30 && step=2 ;;
    'Matting') epochs=150 && val_last=50 && step=5 ;;
esac

# Train
devices=$2
nproc_per_node=$(echo ${devices%%,} | grep -o "," | wc -l)

to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)
if [ ${to_be_distributed} == "True" ]
then
    # Adapt the nproc_per_node by the number of GPUs. Give 8989 as the default value of master_port.
    echo "Multi-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --standalone --nproc_per_node $((nproc_per_node+1)) \
    train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
        --dist ${to_be_distributed} \
        --resume ../ckpt-general/general/epoch_170.pth \
        --use_accelerate
else
    echo "Single-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    python train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
        --dist ${to_be_distributed} \
        --resume xx/xx-epoch_244.pth \
        --use_accelerate
fi

echo Training finished at $(date)
