#!/bin/sh

# 0a. Perform cleanup
rm -rf SM-FLWR_Orchestrator_*

# 0b. Update climateRL environments
# cd fedrl-climate-envs && pip install . && cd ..

# 1a. Function to display usage
usage() {
    echo "Usage: $0 --tag <tag> --env_id <env_id>"
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

# 2. Define the base directory
BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

# 3. List of algorithms
ALGOS=("ddpg" "td3")

# 4. Get the current date and time in YYYY-MM-DD_HH-MM format
NOW=$(date +%F_%H-%M)

# 5. Loop through each algorithm and execute the script
for ALGO in "${ALGOS[@]}"; do
    WANDB_GROUP="${TAG}_${NOW}"
    # Submit each algorithm run as a separate Slurm job
    sbatch slurm_smartsim.sh --rl_algo $ALGO --env_id $ENV_ID --tag $TAG --wandb_group $WANDB_GROUP
done
