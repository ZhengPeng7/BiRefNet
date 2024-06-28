devices=${1:-0}
pred_root=${2:-e_preds}

# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root}

echo Inference finished at $(date)

# Evaluation
log_dir=e_logs && mkdir ${log_dir}

task=$(python3 config.py)
case "${task}" in
    "DIS5K") testsets='DIS-VD,DIS-TE1,DIS-TE2,DIS-TE3,DIS-TE4' ;;
    "COD") testsets='CHAMELEON,NC4K,TE-CAMO,TE-COD10K' ;;
    "HRSOD") testsets='DAVIS-S,TE-HRSOD,TE-UHRSD,DUT-OMRON,TE-DUTS' ;;
    "DIS5K+HRSOD+HRS10K") testsets='DIS-VD' ;;
    "P3M-10k") testsets='TE-P3M-500-P,TE-P3M-500-NP' ;;
esac
testsets=(`echo ${testsets} | tr ',' ' '`) && testsets=${testsets[@]}

for testset in ${testsets}; do
    nohup python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testset} > ${log_dir}/eval_${testset}.out 2>&1 &
done


echo Evaluation started at $(date)
