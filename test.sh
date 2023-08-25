devices=$1

# Inference
CUDA_VISIBLE_DEVICES=${devices} python inference.py

echo Inference is finished at $(date)

# Evaluation
python eval_existingOnes.py \
    # --metrics S+MAE+E+F+WF  \

echo Evaluation is finished at $(date)
