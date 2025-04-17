#!/bin/sh

BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

runs=("$BASE_DIR"/baselines/optuna/results/*v*)

for run in "${runs[@]}"; do
    exp_id=$(basename "$run")
    echo "Reading $run ..."
    python $BASE_DIR/baselines/optuna/results/read.py --exp_id $exp_id
done
