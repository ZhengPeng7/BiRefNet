devices=${1:-0}
pred_root=${2:-e_preds}

# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root}

echo Inference finished at $(date)

# Evaluation
log_dir=e_logs
mkdir ${log_dir}

testsets=DIS-VD  && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=DIS-TE1 && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=DIS-TE2 && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=DIS-TE3 && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
testsets=DIS-TE4 && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &

# testsets=CHAMELEON  && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
# testsets=NC4K && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
# testsets=TE-CAMO && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
# testsets=TE-COD10K && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &

# testsets=DAVIS-S  && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
# testsets=TE-HRSOD  && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
# testsets=TE-UHRSD && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
# testsets=DUT-OMRON && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &
# testsets=TE-DUTS && nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testsets} > ${log_dir}/eval_${testsets}.out 2>&1 &

echo Evaluation started at $(date)
