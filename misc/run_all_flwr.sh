BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

# Perform cleanup
# rm -rf SM-FLWR/*

# Update climateRL environments
# cd fedrl-climate-envs && pip install . && cd ..

# ebm-v3
# n_clients = 6

# flwr_actor ONLY
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v3-optim-L-20k-a6-fed05" --env_id "EnergyBalanceModel-v3" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 5
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v3-optim-L-20k-a6-fed10" --env_id "EnergyBalanceModel-v3" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 10
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v3-optim-L-20k-a6-nofed" --env_id "EnergyBalanceModel-v3" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 125  # max: 100

# ebm-v3
# n_clients = 2

# flwr_actor ONLY
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v3-optim-L-20k-a2-fed05" --env_id "EnergyBalanceModel-v3" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 5
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v3-optim-L-20k-a2-fed10" --env_id "EnergyBalanceModel-v3" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 10
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v3-optim-L-20k-a2-nofed" --env_id "EnergyBalanceModel-v3" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 125  # max: 100

# # ------

# ebm-v2
# n_clients = 6

# flwr_actor ONLY
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v2-optim-L-20k-a6-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 5
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v2-optim-L-20k-a6-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 10
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v2-optim-L-20k-a6-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 6 --flwr_episodes 125  # max: 100

# ebm-v2
# n_clients = 2

# flwr_actor ONLY
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v2-optim-L-20k-a2-fed05" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 5
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v2-optim-L-20k-a2-fed10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 10
# source "$BASE_DIR/smartsim/run_distributed.sh" --tag "ebm-v2-optim-L-20k-a2-nofed" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --num_clients 2 --flwr_episodes 125  # max: 100

# # ------

# ebm-v3 - inference

# source "$BASE_DIR/misc/run_inference_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a6-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 6
# source "$BASE_DIR/misc/run_inference_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a6-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 6
# source "$BASE_DIR/misc/run_inference_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a6-nofed" --optim_group "ebm-v1-optim-L-20k" --num_clients 6

# source "$BASE_DIR/misc/run_inference_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a2-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 2
# source "$BASE_DIR/misc/run_inference_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a2-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 2
# source "$BASE_DIR/misc/run_inference_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a2-nofed" --optim_group "ebm-v1-optim-L-20k" --num_clients 2

# source "$BASE_DIR/misc/run_inference_global_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a6-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 6
# source "$BASE_DIR/misc/run_inference_global_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a6-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 6

# source "$BASE_DIR/misc/run_inference_global_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a2-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 2
# source "$BASE_DIR/misc/run_inference_global_flwr_v3.sh" --env_id "EnergyBalanceModel-v3" --tag "ebm-v3-optim-L-20k-a2-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 2

# # ------

# ebm-v2 - inference

# source "$BASE_DIR/misc/run_inference_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a6-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 6
# source "$BASE_DIR/misc/run_inference_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a6-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 6
# source "$BASE_DIR/misc/run_inference_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a6-nofed" --optim_group "ebm-v1-optim-L-20k" --num_clients 6

# source "$BASE_DIR/misc/run_inference_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a2-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 2
# source "$BASE_DIR/misc/run_inference_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a2-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 2
# source "$BASE_DIR/misc/run_inference_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a2-nofed" --optim_group "ebm-v1-optim-L-20k" --num_clients 2

# source "$BASE_DIR/misc/run_inference_global_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a6-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 6
# source "$BASE_DIR/misc/run_inference_global_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a6-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 6

# source "$BASE_DIR/misc/run_inference_global_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a2-fed05" --optim_group "ebm-v1-optim-L-20k" --num_clients 2
# source "$BASE_DIR/misc/run_inference_global_flwr_v2.sh" --env_id "EnergyBalanceModel-v2" --tag "ebm-v2-optim-L-20k-a2-fed10" --optim_group "ebm-v1-optim-L-20k" --num_clients 2
