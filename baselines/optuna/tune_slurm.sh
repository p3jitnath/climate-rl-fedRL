#!/bin/sh

#SBATCH --job-name=pn341_optuna_climlab
#SBATCH --output=/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl/slurm/optuna_climlab_%j.out
#SBATCH --error=/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl/slurm/optuna_climlab_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --account=ai4er
#SBATCH --partition=highres
#SBATCH --qos=highres

BASE_DIR=/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl

set -x

# 1a. Function to display usage
usage() {
    echo "Usage: sbatch $1 --exp_id <exp_id> --env_id <env_id>"
    exit 1
}

# 1b. Check if no arguments were passed
if [ "$#" -eq 0 ]; then
    usage
fi

# 1c. Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_id)
            EXP_ID="$2"
            shift 2
            ;;
        --env_id)
            ENV_ID="$2"
            shift 2
            ;;
        *) # Handle unknown option
            usage
            ;;
    esac
done

# 1d: Print parsed values (for debugging)
echo "exp_id: $EXP_ID"
echo "env_id: $ENV_ID"

# 1e. Check if all flags are set
if [ -z "$EXP_ID" ] || [ -z "$ENV_ID" ]; then
    echo "Error: All flags are required."
    usage
fi

# __doc_head_address_start__

# Checking the conda environment
echo "PYTHON: $(which python)"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# If we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# __doc_head_address_end__

port=$(shuf -i 6380-6580 -n 1)

k=$(shuf -i 20-55 -n 1)
min_port=$((k * 1000))
max_port=$((min_port + 999))

ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --min-worker-port $min_port --max-worker-port $max_port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --include-dashboard=False --num-gpus 0 --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 30

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --min-worker-port $min_port --max-worker-port $max_port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 0 --block &
    sleep 30
done

python -u $BASE_DIR/baselines/optuna/tune.py --exp_id $EXP_ID --env_id $ENV_ID
