#!/bin/sh

#SBATCH --job-name=pn341_smartsim_slurm
#SBATCH --output=/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl/slurm/ray_slurm_%j.out
#SBATCH --error=/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl/slurm/ray_slurm_%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --account=ai4er
#SBATCH --partition=highres
#SBATCH --qos=highres

BASE_DIR=/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl
LOG_DIR="$BASE_DIR/slurm"

set -x

# __doc_head_address_start__

# checking the conda environment
echo "PYTHON: $(which python)"

# getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPv6 address detected. We split the IPv4 address as $head_node_ip"
fi

# __doc_head_address_end__

port=$(shuf -i 6380-6580 -n 1)

k=$(shuf -i 20-55 -n 1)
min_port=$((k * 1000))
max_port=$((min_port + 999))

redis_port=$(shuf -i 6280-6480 -n 1)
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

export DISTRIBUTED=1
export SSDB=$head_node_ip:$redis_port
echo "SSDB: $SSDB"


echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --min-worker-port $min_port --max-worker-port $max_port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --include-dashboard=False --block & \
    --output="$LOG_DIR/ray_slurm_${SLURM_JOB_ID}.out" \
    --error="$LOG_DIR/ray_slurm_${SLURM_JOB_ID}.err"

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
        --num-cpus "${SLURM_CPUS_PER_TASK}" --block & \
        --output="$LOG_DIR/ray_slurm_${SLURM_JOB_ID}.out" \
        --error="$LOG_DIR/ray_slurm_${SLURM_JOB_ID}.err"
    sleep 30
done

# Running the test smartsim script
python -u $BASE_DIR/run_smartsim.py
