#!/bin/sh

# Base directory
BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

# List of algorithms
# ALGOS=("ddpg" "dpg" "td3" "reinforce" "trpo" "ppo" "sac" "tqc" "avg")
ALGOS=("ddpg" "td3" "tqc")

# List of expirements
EXPERIMENT_IDS=("ebm-v2-optim-L-20k-a2-fed05" "ebm-v2-optim-L-20k-a2-fed10" "ebm-v2-optim-L-20k-a2-nofed" "ebm-v2-optim-L-20k-a6-fed05" "ebm-v2-optim-L-20k-a6-fed10" "ebm-v2-optim-L-20k-a6-nofed")

# Loop through each run
for EXPERIMENT_ID in "${EXPERIMENT_IDS[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
        for SEED in {1..10}; do
            echo "Preparing run — ID: $EXPERIMENT_ID | Algo: $ALGO | Seed: $SEED"

            # Submit the job to SLURM
            sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=results_flwr_merge_$EXPERIMENT_ID_$ALGO_$SEED
#SBATCH --output=$BASE_DIR/slurm/results_flwr_merge_$EXPERIMENT_ID_$ALGO_$SEED%j.out
#SBATCH --error=$BASE_DIR/slurm/results_flwr_merge_$EXPERIMENT_ID_$ALGO_$SEED%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --account=ai4er
#SBATCH --partition=standard
#SBATCH --qos=high

conda activate venv
python $BASE_DIR/misc/merge_flwr.py --exp_id $EXPERIMENT_ID --algo $ALGO --seed $SEED
EOT
        done
    done
done

echo "All jobs submitted to SLURM."
