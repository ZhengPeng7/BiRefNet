#!/bin/bash
# Run script
method="$1"
epochs=80
val_last=20
step=20
testsets=DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4

# Train
devices=$2
nproc_per_node=$(echo ${devices%%,} | grep -o "," | wc -l)

to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

if [ ${to_be_distributed} == "True" ]
then
    # Adapt the nproc_per_node by the number of GPUs. Give 29500 as the default value of master_port.
    echo "Multi-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --nproc_per_node $((nproc_per_node+1)) --master_port=$((29500+${3:-0})) \
    train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
        --testsets ${testsets} \
        --dist ${to_be_distributed}
else
    echo "Single-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    python train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
        --testsets ${testsets} \
        --dist ${to_be_distributed} \
        --resume ckpt/swin_rs1024/ep150.pth
fi

echo Finished at $(date)
