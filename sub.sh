method=$1
devices=$2

srun --nodes=1 --nodelist=Slave1,Slave2,Slave3,Slave4,Slave5 \
--ntasks-per-node=1 \
--gres=gpu:$(($(echo ${devices%%,} | grep -o "," | wc -l)+1)) \
--cpus-per-task=32 \
bash go.sh ${method} ${devices}

hostname
