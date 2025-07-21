BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

# ebm-v2

# flwr_actor ONLY
source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k-a-5" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --flwr_episodes 5
# source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k-a-10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --flwr_episodes 10
# source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k-a-1000" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics false --flwr_episodes 1000

# flwr_critic ONLY
# source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k-c-5" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor false --flwr_critics true --flwr_episodes 5
# source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k-c-10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor false --flwr_critics true --flwr_episodes 10
# source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k-c-1000" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor false --flwr_critics true --flwr_episodes 1000

# flwr_actor and flwr_critic
# source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k-ac-5" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics true --flwr_episodes 5
# source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k-ac-10" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics true --flwr_episodes 10
# source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k-ac-1000" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k" --flwr_actor true --flwr_critics true --flwr_episodes 1000
