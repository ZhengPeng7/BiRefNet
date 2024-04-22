
# Example: ./sub.sh tmp_proj 0,1,2,3 3 --> Use 0,1,2,3 for training, release GPUs, use GPU:3 for inference.

method=${1:-"BSL"}
devices=${2:-0}

# srun --nodes=1 --nodelist=Master,Slave1,Slave2,Slave3,Slave4,Slave5 \
# --ntasks-per-node=1 \
# --gres=gpu:$(($(echo ${devices%%,} | grep -o "," | wc -l)+1)) \
# --cpus-per-task=32 \
bash train.sh ${method} ${devices}

hostname

devices_test=${3:-0}
bash test.sh ${devices_test}
