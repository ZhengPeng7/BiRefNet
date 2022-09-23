#!/bin/bash

# Val
rm -r tmp*

# Train
rm slurm*
rm -r ckpt
rm *.out

# Eval
rm -r evaluation/eval-*
