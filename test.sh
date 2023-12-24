devices=${1:-0}
pred_root=${2:-e_preds}

# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root}

echo Inference is finished at $(date)

# Evaluation
log_dir=e_logs
mkdir ${log_dir}
testsets=DIS-VD  && nohup python eval_existingOnes.py --pred_root ${pred_root} \
    --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=DIS-TE1 && nohup python eval_existingOnes.py --pred_root ${pred_root} \
    --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=DIS-TE2 && nohup python eval_existingOnes.py --pred_root ${pred_root} \
    --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=DIS-TE3 && nohup python eval_existingOnes.py --pred_root ${pred_root} \
    --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=DIS-TE4 && nohup python eval_existingOnes.py --pred_root ${pred_root} \
    --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &

echo Evaluation is started at $(date)
