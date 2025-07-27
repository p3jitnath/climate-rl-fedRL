#!/bin/sh

# 1a. Function to display usage
usage() {
    echo "Usage: $0 --tag <tag> --env_id <env_id> --optim_group <optim_group> --num_clients <num_clients>"
    exit 1
}

# 1b. Check if no arguments were passed
if [ "$#" -eq 0 ]; then
    usage
fi

# 1c. Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tag) # Extract the tag value
            TAG="$2"
            shift 2
            ;;
        --env_id) # Extract the env_id value
            ENV_ID="$2"
            shift 2
            ;;
        --optim_group) # Extract the optim_group value
            OPTIM_GROUP="$2"
            shift 2
            ;;
        --num_clients) # Extract the num_clients value
            NUM_CLIENTS="$2"
            shift 2
            ;;
        *) # Handle unknown option
            usage
            ;;
    esac
done

# 1d. Check if TAG is set
if [ -z "$TAG" ]; then
    echo "Error: Tag is required."
    usage
fi

# 1e. Check if ENV_ID is set
if [ -z "$ENV_ID" ]; then
    echo "Error: Environment id is required."
    usage
fi

# 1f. Check if OPTIM_GROUP is set
if [ -z "$OPTIM_GROUP" ]; then
    echo "Error: Optimisation group is required."
    usage
fi

# 1f. Check if NUM_CLIENTS is set
if [ -z "$NUM_CLIENTS" ]; then
    echo "Error: Number of clients is required."
    usage
fi

# 2. Define the base directory
BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

# 3. List of algorithms
# ALGOS=("ddpg" "dpg" "td3" "reinforce" "trpo" "ppo" "sac" "avg")
# ALGOS=("tqc")
ALGOS=("ddpg" "td3")

# 4. Get the current date and time in YYYY-MM-DD_HH-MM format
NOW=$(date +%F_%H-%M)
# NOW=$(basename $(find ${BASE_DIR}/runs/ -maxdepth 1 -type d -name "infxG_${TAG}_*" | grep -E "${TAG}_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}$" | sort -r | head -n 1) | sed -E "s/^infxG_${TAG}_//")
WANDB_GROUP="infxG_${TAG}_${NOW}"
echo $WANDB_GROUP

# 5. Loop through each algorithm and execute the script
for ALGO in "${ALGOS[@]}"; do
    for SEED in {1..10}; do
        for FLWR_CLIENT in $(seq 0 $((NUM_CLIENTS - 1))); do
            # Submit each algorithm run as a separate Slurm job
        sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=pn341_${ALGO}_${TAG}_${FLWR_CLIENT}
#SBATCH --output=$BASE_DIR/slurm/infxG_${ALGO}_${TAG}_${SEED}_${FLWR_CLIENT}_%j.out
#SBATCH --error=$BASE_DIR/slurm/infxG_${ALGO}_${TAG}_${SEED}_${FLWR_CLIENT}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --account=ai4er
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --nodelist=host[1201-1272]

## SBATCH --account=ai4er
## SBATCH --partition=standard
## SBATCH --qos=high
## SBATCH --nodelist=host[1201-1272]

## SBATCH --account=orchid
## SBATCH --partition=orchid
## SBATCH --qos=orchid
## SBATCH --gres=gpu:1

conda activate venv
cd "$BASE_DIR"
export WANDB_MODE=offline
export INFERENCE=1
export NUM_CLIENTS=$NUM_CLIENTS

python -u "$BASE_DIR/fedrl/inference_global.py" --env_id "$ENV_ID" --algo $ALGO --optim_group "$OPTIM_GROUP" --wandb_group "$WANDB_GROUP" --flwr_client $FLWR_CLIENT --seed $SEED --capture_video --num_steps 200 --record_step 20000
EOT
        # sleep 60
        done
    done
done
