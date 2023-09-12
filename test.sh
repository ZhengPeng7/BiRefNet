devices=${1:-7}
pred_root=${2:-e_preds}

# Inference
log_dir=e_logs
mkdir ${log_dir}

CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root}

# testsets=DIS-VD  && CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root} \
#     --testsets ${testsets}
# testsets=DIS-TE1 && CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root} \
#     --testsets ${testsets}
# testsets=DIS-TE2 && CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root} \
#     --testsets ${testsets}
# testsets=DIS-TE3 && CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root} \
#     --testsets ${testsets}
# testsets=DIS-TE4 && CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root} \
#     --testsets ${testsets}
# # Select testsets from: DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4, COD10K+NC4K+CAMO+CHAMELEON, DAVIS-S+HRSOD-TE+UHRSD-TE+DUTS-TE+DUT-OMRON

echo Inference is finished at $(date)

# Evaluation
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

echo Evaluation is finished at $(date)
