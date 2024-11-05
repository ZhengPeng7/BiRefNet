#!/bin/bash
rm -rf __pycache__ */__pycache__ */*/__pycache__

# Val
rm -r tmp*

# Train
rm slurm*
rm -r ckpt
rm nohup.out*
rm nohup.log*

# Eval
rm -r evaluation/eval-*
rm -r tmp*
rm -r e_logs/

# System
rm core-*-python-*

# Inference cache
rm -rf images_todo/
rm -rf predictions/

clear
