#!/bin/bash

current_date_time="`date +%Y-%m-%d-%H:%M:%S`"
echo "current_date_time: $current_date_time"



export OMP_NUM_THREADS=8


WORKSPACE=$(dirname "$PWD")
ENV_PATH="/home/zhanyi/anaconda3"
PYTHON="$ENV_PATH/bin/python"
echo "Using $PYTHON"

export "PYTHONPATH=$WORKSPACE:$PYTHONPATH"


$PYTHON train_main.py \
    --log_name="train_${current_date_time}.log" \
    | tee "train_${current_date_time}.log"


