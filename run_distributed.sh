#!/bin/sh

# 0a. Perform cleanup
rm -rf SM-FLWR_Orchestrator_*

# 0b. Update climateRL environments
cd fedrl-climate-envs && pip install . && cd ..

# 1a. Function to display usage
usage() {
    echo "Usage: $0 --tag <tag> --env_id <env_id> [--optim_group <optim_group>] [--flwr_actor <true|false>] [--flwr_critics <true|false>]"
    exit 1
}

# 1b. Check if no arguments were passed
if [ "$#" -eq 0 ]; then
    usage
fi

# 1c. Set default values
FLWR_ACTOR=true
FLWR_CRITICS=false
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
ALGOS=("ddpg" "dpg" "td3" "reinforce" "trpo" "ppo" "sac" "tqc" "avg")

# 4. Get the current date and time in YYYY-MM-DD_HH-MM format
NOW=$(date +%F_%H-%M)
WANDB_GROUP="${TAG}_${NOW}"

# 5. Loop through each algorithm and execute the script
for ALGO in "${ALGOS[@]}"; do
    # Submit each algorithm run as a separate Slurm job
    sbatch slurm_smartsim.sh \
           --rl_algo   "$ALGO" \
           --env_id    "$ENV_ID" \
           ${OPTIM_GROUP:+--optim_group "$OPTIM_GROUP"} \
           --tag "$TAG" \
           --wandb_group  "$WANDB_GROUP" \
           --flwr_actor   "$FLWR_ACTOR" \
           --flwr_critics "$FLWR_CRITICS" \
           --seed 0
done
