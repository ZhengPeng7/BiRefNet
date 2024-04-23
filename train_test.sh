#!/bin/sh

method=${1:-"BSL"}
devices=${2:-0}

bash train.sh ${method} ${devices}

devices_test=${3:-0}
bash test.sh ${devices_test}

hostname
