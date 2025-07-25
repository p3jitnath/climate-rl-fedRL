BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

# Perform cleanup
# rm -rf SM-FLWR/*

# Update climateRL environments
# cd fedrl-climate-envs && pip install . && cd ..

# ebm-v2
# n_clients = 6

# flwr_actor ONLY
source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-a6-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 5
source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-a6-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 10
source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-a6-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 111  # max: 100

# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-homo-64L-a6-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-homo-64L" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 5
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-homo-64L-a6-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-homo-64L" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 10
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-homo-64L-a6-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-homo-64L" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 111  # max: 100


# ebm-v2
# n_clients = 2

flwr_actor ONLY
source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-a2-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 5
source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-a2-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 10
source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-a2-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 111  # max: 100

# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-homo-64L-a2-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-homo-64L" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 5
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-homo-64L-a2-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-homo-64L" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 10
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-homo-64L-a2-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-homo-64L" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 111  # max: 100


# # ------

# ebm-v2 - inference

# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a6-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 6
# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a6-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 6
# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a6-nofed" --optim_group "ebm-v1-optim-L-20k" --num_clients 6

# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a2-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 2
# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a2-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 2
# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a2-nofed" --optim_group "ebm-v1-optim-L-20k" --num_clients 2

# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-homo-64L-a6-fed05" --optim_group "ebm-v1-homo-64L" --num_clients 6
# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-homo-64L-a6-fed10" --optim_group "ebm-v1-homo-64L" --num_clients 6
# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-homo-64L-a6-nofed" --optim_group "ebm-v1-homo-64L" --num_clients 6

# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-homo-64L-a2-fed05" --optim_group "ebm-v1-homo-64L" --num_clients 2
# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-homo-64L-a2-fed10" --optim_group "ebm-v1-homo-64L" --num_clients 2
# source "$BASE_DIR/misc/run_inference_flwr.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-homo-64L-a2-nofed" --optim_group "ebm-v1-homo-64L" --num_clients 2


# # ------ archive ------

# ebm-v2
# n_clients = 6

# flwr_critic ONLY
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-c6-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor false --flwr_critics true --num_clients 6 --flwr_episodes 5
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-c6-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor false --flwr_critics true --num_clients 6 --flwr_episodes 10
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-c6-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor false --flwr_critics true --num_clients 6 --flwr_episodes 111  # max: 100

# flwr_actor and flwr_critic
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-ac6-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics true --num_clients 6 --flwr_episodes 5
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-ac6-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics true --num_clients 6 --flwr_episodes 10
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-ac6-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics true --num_clients 6 --flwr_episodes 111  # max: 100

# ebm-v2
# n_clients = 2

# flwr_critic ONLY
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-c2-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor false --flwr_critics true --num_clients 2 --flwr_episodes 5
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-c2-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor false --flwr_critics true --num_clients 2 --flwr_episodes 10
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-c2-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor false --flwr_critics true --num_clients 2 --flwr_episodes 111  # max: 100

# flwr_actor and flwr_critic
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-ac2-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics true --num_clients 2 --flwr_episodes 5
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-ac2-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics true --num_clients 2 --flwr_episodes 10
# source "$BASE_DIR/run-distributed.sh" --tag "ebm-v2-optim-L-20k-ac2-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics true --num_clients 2 --flwr_episodes 111  # max: 100
