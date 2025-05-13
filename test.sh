devices=${1:-0}
pred_root=${2:-e_preds}
resolutions=${3:-"config.size"}

# Inference
# resolutions="1024x1024 None"
for resolution in ${resolutions}; do
    CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root} --resolution ${resolution}
done

echo Inference finished at $(date)

# Evaluation
log_dir=e_logs && mkdir ${log_dir}

task=$(python3 config.py --print_task)
testsets=$(python3 config.py --print_testsets)

testsets=(`echo ${testsets} | tr ',' ' '`) && testsets=${testsets[@]}

for testset in ${testsets}; do
    # python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testset} > ${log_dir}/eval_${testset}.out
    nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testset} > ${log_dir}/eval_${testset}.out 2>&1 &
done


echo Evaluation started at $(date)
