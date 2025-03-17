import json

from gymnasium.envs.registration import register

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

register(
    id="SimpleClimateBiasCorrection-v0",
    entry_point="fedrl_climate_envs.envs:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)

register(
    id="EnergyBalanceModel-v0",
    entry_point="fedrl_climate_envs.envs:EnergyBalanceModelEnv",
    max_episode_steps=200,
)

register(
    id="EnergyBalanceModel-v1",
    entry_point="fedrl_climate_envs.envs.energy_balance_model_v1:EnergyBalanceModelEnv",
    max_episode_steps=200,
)

register(
    id="EnergyBalanceModel-v2",
    entry_point="fedrl_climate_envs.envs.energy_balance_model_v2:EnergyBalanceModelEnv",
    max_episode_steps=200,
)

register(
    id="EnergyBalanceModel-v3",
    entry_point="fedrl_climate_envs.envs.energy_balance_model_v3:EnergyBalanceModelEnv",
    max_episode_steps=200,
)
