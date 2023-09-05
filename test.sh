devices=${1:-7}
pred_root=${2:-e_preds}

# Inference
CUDA_VISIBLE_DEVICES=${devices} python inference.py \
    --pred_root ${pred_root} \
    # --testsets COD10K
# Select testsets from: DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4, COD10K+NC4K+CAMO+CHAMELEON, DAVIS-S+HRSOD-TE+UHRSD-TE+DUTS-TE+DUT-OMRON

echo Inference is finished at $(date)

# Evaluation
python eval_existingOnes.py \
    --pred_root ${pred_root} \
    # --metrics S+MAE  \

echo Evaluation is finished at $(date)
