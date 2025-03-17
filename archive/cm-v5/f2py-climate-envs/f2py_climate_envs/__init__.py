import json

from gymnasium.envs.registration import register

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl/cm-v5"

register(
    id="SimpleClimateBiasCorrection-v0",
    entry_point="fedrl_climate_envs.envs:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)
