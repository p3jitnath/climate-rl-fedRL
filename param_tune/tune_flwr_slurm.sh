#!/bin/sh

#SBATCH --job-name=pn341_ray_slurm_optimise_orchestrator
#SBATCH --output=/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl/slurm/ray_slurm_flwr_orchestrator_%j.out
#SBATCH --error=/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl/slurm/ray_slurm_flwr_orchestrator_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --account=ai4er
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --nodelist=host[1201-1272]

source ~/miniconda3/etc/profile.d/conda.sh && conda activate venv
BASE_DIR=/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl
set -x

# 1a. Function to display usage
usage() {
    echo "Usage: sbatch tune_flwr_slurm.sh --algo <algo> --exp_id <exp_id> --env_id <env_id> --opt_timesteps <steps>"
    exit 1
}

# 1b. Check if no arguments were passed
if [ "$#" -eq 0 ]; then
    usage
fi

# 1c. Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --algo)
            ALGO="$2"
            shift 2
            ;;
        --exp_id)
            EXP_ID="$2"
            shift 2
            ;;
        --env_id)
            ENV_ID="$2"
            shift 2
            ;;
        --opt_timesteps)
            OPT_TIMESTEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

# 1d: Print parsed values (for debugging)
echo "algo: $ALGO"
echo "exp_id: $EXP_ID"
echo "env_id: $ENV_ID"
echo "opt_timesteps: $OPT_TIMESTEPS"

# 1e. Check if all flags are set
if [ -z "$ALGO" ] || [ -z "$EXP_ID" ] || [ -z "$ENV_ID" ] || [ -z "$OPT_TIMESTEPS" ] ; then
    echo "Error: All flags are required."
    usage
fi

# checking the conda environment
echo "PYTHON: $(which python)"

python -u $BASE_DIR/param_tune/tune_flwr.py \
        --algo $ALGO \
        --exp_id $EXP_ID \
        --env_id $ENV_ID \
        --opt_timesteps $OPT_TIMESTEPS
