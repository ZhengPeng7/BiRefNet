#!/bin/bash
rm -rf __pycache__ */__pycache__

# Val
rm -r tmp*

# Train
rm slurm*
rm -r ckpt
rm nohup.out*

# Eval
rm -r evaluation/eval-*
rm -r tmp*
rm -r e_logs/

# System
rm core-*-python-*

clear
