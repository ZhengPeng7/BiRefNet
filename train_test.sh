#!/bin/sh
# Example: `setsid nohup ./train_test.sh BiRefNet_tmp 0,1,2,3,4,5,6,7 0 &>nohup.log &`

method=${1:-"BSL"}
devices=${2:-"0,1,2,3,4,5,6,7"}

bash train.sh ${method} ${devices}

devices_test=${3:-0}
bash test.sh ${devices_test}

hostname
