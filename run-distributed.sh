#!/bin/sh

# 1a. Function to display usage
usage() {
    echo "Usage: $0 --tag <tag> --env_id <env_id> [--optim_group <optim_group>] [--flwr_actor <true|false>] [--flwr_critics <true|false>] [--flwr_episodes <flwr_episodes>] [--num_clients <num_clients>]"
    exit 1
}

# 1b. Check if no arguments were passed
if [ "$#" -eq 0 ]; then
    usage
fi

# 1c. Set default values
FLWR_ACTOR=true
FLWR_CRITICS=false
FLWR_EPISODES=5
NUM_CLIENTS=2
OPTIM_GROUP=""

# 1d. Parse command-line arguments
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
        --flwr_actor) # Extract the flwr_actor bool value
            FLWR_ACTOR="$2"
            shift 2
            ;;
        --flwr_critics) # Extract the flwr_critics bool value
            FLWR_CRITICS="$2"
            shift 2
            ;;
        --flwr_episodes) # Extract the flwr_episodes value
            FLWR_EPISODES="$2"
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

# 1e. Check if TAG is set
if [ -z "$TAG" ]; then
    echo "Error: Tag is required."
    usage
fi

# 1f. Check if ENV_ID is set
if [ -z "$ENV_ID" ]; then
    echo "Error: Environment id is required."
    usage
fi

# 2. Define the base directory
BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

# 3. List of algorithms
# ALGOS=("ddpg" "dpg" "td3" "reinforce" "trpo" "ppo" "sac" "avg")
# ALGOS=("tqc")
ALGOS=("ddpg" "td3")

# 4. Get the current date and time in YYYY-MM-DD_HH-MM format
# NOW=$(date +%F_%H-%M)
NOW=$(basename $(find ${BASE_DIR}/runs/ -maxdepth 1 -type d -name "${TAG}_*" | grep -E "${TAG}_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}$" | sort -r | head -n 1) | sed -E "s/^${TAG}_//")
WANDB_GROUP="${TAG}_${NOW}"
echo $WANDB_GROUP

# 5. Loop through each algorithm and execute the script
for ALGO in "${ALGOS[@]}"; do
    for SEED in {1..10}; do
        # Submit each algorithm run as a separate Slurm job
        sbatch slurm_smartsim.sh \
            --rl_algo   "$ALGO" \
            --env_id    "$ENV_ID" \
            ${OPTIM_GROUP:+--optim_group "$OPTIM_GROUP"} \
            --tag "$TAG" \
            --wandb_group  "$WANDB_GROUP" \
            --flwr_actor   "$FLWR_ACTOR" \
            --flwr_critics "$FLWR_CRITICS" \
            --flwr_episodes "$FLWR_EPISODES" \
            --num_clients "$NUM_CLIENTS" \
            --seed $SEED
    done
done
