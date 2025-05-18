BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

source "$BASE_DIR/run_distributed.sh" --tag "ebm-v2-optim-L-20k" --env_id "EnergyBalanceModel-v2" --optim_group "ebm-v1-optim-L-20k"
source "$BASE_DIR/run_distributed.sh" --tag "ebm-v3-optim-L-20k" --env_id "EnergyBalanceModel-v3" --optim_group "ebm-v1-optim-L-20k"
