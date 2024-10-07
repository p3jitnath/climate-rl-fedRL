import json

from gymnasium.envs.registration import register

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-f2py/cm-v3"

register(
    id="SimpleClimateBiasCorrection-v0",
    entry_point="f2py_climate_envs.envs:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)
