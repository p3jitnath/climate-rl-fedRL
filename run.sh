# 0. Perform cleanup
# Stop all pending ray processes

# ray stop --force
rm -rf SM-FLWR_Orchestrator_*

export SSDB=localhost:6380
export DISTRIBUTED=0

# # Check for keys matching "SIG*" pattern in Redis on port 6380
# sig_keys=$(redis-cli -p 6380 KEYS "SIG*")
# # Check for keys matching "actor*" pattern in Redis on port 6380
# actor_keys=$(redis-cli -p 6380 KEYS "actor*")
# # Check for keys matching "py2f*" pattern in Redis on port 6380
# py2f_keys=$(redis-cli -p 6380 KEYS "py2f*")
# # Check for keys matching "f2py*" pattern in Redis on port 6380
# f2py_keys=$(redis-cli -p 6380 KEYS "f2py*")

# # If there are SIG* keys, delete them
# if [[ -n "$sig_keys" ]]; then
#   echo "Found SIG* keys: $sig_keys"
#   echo "$sig_keys" | xargs redis-cli -p 6380 DEL
#   echo "SIG* keys deleted."
# else
#   echo "No keys found matching pattern 'SIG*'."
# fi

# # If there are actor* keys, delete them
# if [[ -n "$actor_keys" ]]; then
#   echo "Found actor* keys: $actor_keys"
#   echo "$actor_keys" | xargs redis-cli -p 6380 DEL
#   echo "actor* keys deleted."
# else
#   echo "No keys found matching pattern 'actor*'."
# fi

# # If there are py2f* keys, delete them
# if [[ -n "$py2f_keys" ]]; then
#   echo "Found py2f* keys: $py2f_keys"
#   echo "$py2f_keys" | xargs redis-cli -p 6380 DEL
#   echo "py2f* keys deleted."
# else
#   echo "No keys found matching pattern 'py2f*'."
# fi

# # If there are f2py* keys, delete them
# if [[ -n "$f2py_keys" ]]; then
#   echo "Found f2py* keys: $f2py_keys"
#   echo "$f2py_keys" | xargs redis-cli -p 6380 DEL
#   echo "f2py* keys deleted."
# else
#   echo "No keys found matching pattern 'f2py*'."
# fi

# 1. Set experiment environment variables
export RL_ALGO="ddpg"
export ENV_ID="EnergyBalanceModel-v3"
export OPTIM_GROUP="ebm-v1-optim-L-20k"
export WANDB_GROUP="test"
export FLWR_ACTOR="true"
export FLWR_CRITICS="false"
export SEED="0"

# 1. Run smartsim
cd fedrl-climate-envs && pip install . && cd ..
python run_smartsim.py
