#!/bin/sh

BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

runs=("$BASE_DIR"/param_tune/results/*v*)

for run in "${runs[@]}"; do
    exp_id=$(basename "$run")
    echo "Reading $run ..."
    if [[ "$run" == *fedRL* ]]; then
        python $BASE_DIR/param_tune/results/read_flwr.py --exp_id $exp_id
    else
        python $BASE_DIR/param_tune/results/read.py --exp_id $exp_id
    fi
done
