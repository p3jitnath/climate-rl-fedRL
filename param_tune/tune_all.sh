BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

# ebm-v0

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "ebm-v0-optim-L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200  && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "ebm-v0-optim-L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "ebm-v0-optim-L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "ebm-v0-optim-L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "ebm-v0-optim-L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "ebm-v0-optim-L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "ebm-v0-optim-L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "ebm-v0-optim-L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "ebm-v0-optim-L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 && sleep 60

# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "ebm-v0-homo-64L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 --homo64  && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "ebm-v0-homo-64L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "ebm-v0-homo-64L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "ebm-v0-homo-64L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "ebm-v0-homo-64L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "ebm-v0-homo-64L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "ebm-v0-homo-64L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "ebm-v0-homo-64L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "ebm-v0-homo-64L" --env_id "EnergyBalanceModel-v0" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60

# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "ebm-v0-optim-L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "ebm-v0-optim-L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "ebm-v0-optim-L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "ebm-v0-optim-L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "ebm-v0-optim-L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "ebm-v0-optim-L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "ebm-v0-optim-L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "ebm-v0-optim-L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "ebm-v0-optim-L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 && sleep 60

# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "ebm-v0-homo-64L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "ebm-v0-homo-64L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "ebm-v0-homo-64L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "ebm-v0-homo-64L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "ebm-v0-homo-64L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "ebm-v0-homo-64L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "ebm-v0-homo-64L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "ebm-v0-homo-64L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "ebm-v0-homo-64L-20k" --env_id "EnergyBalanceModel-v0" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60

# sleep 600

# ebm-v1

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "ebm-v1-optim-L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "ebm-v1-optim-L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "ebm-v1-optim-L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "ebm-v1-optim-L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "ebm-v1-optim-L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "ebm-v1-optim-L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "ebm-v1-optim-L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "ebm-v1-optim-L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "ebm-v1-optim-L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 && sleep 60

# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "ebm-v1-homo-64L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "ebm-v1-homo-64L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "ebm-v1-homo-64L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "ebm-v1-homo-64L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "ebm-v1-homo-64L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "ebm-v1-homo-64L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "ebm-v1-homo-64L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "ebm-v1-homo-64L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "ebm-v1-homo-64L" --env_id "EnergyBalanceModel-v1" --opt_timesteps 10000 --num_steps 200 --homo64 && sleep 60

# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "ebm-v1-optim-L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "ebm-v1-optim-L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "ebm-v1-optim-L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "ebm-v1-optim-L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "ebm-v1-optim-L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "ebm-v1-optim-L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "ebm-v1-optim-L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "ebm-v1-optim-L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "ebm-v1-optim-L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 && sleep 60

# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "ebm-v1-homo-64L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "ebm-v1-homo-64L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "ebm-v1-homo-64L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "ebm-v1-homo-64L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "ebm-v1-homo-64L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "ebm-v1-homo-64L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "ebm-v1-homo-64L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "ebm-v1-homo-64L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "ebm-v1-homo-64L-20k" --env_id "EnergyBalanceModel-v1" --opt_timesteps 20000 --num_steps 200 --homo64 && sleep 60
