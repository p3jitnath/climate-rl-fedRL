import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Optional

import fedrl_climate_envs
import gymnasium as gym
import numpy as np
import ray
import tyro
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
sys.path.append(BASE_DIR)


@dataclass
class Args:
    algo: str = "optuna-tpe"
    """name of the optuna algo"""
    exp_id: str = "ebm-v0"
    """name of the experiment"""
    env_id: str = "EnergyBalanceModel-v0"
    """name of the environment"""
    seed: int = 1
    """seed of the experiment"""


def make_env(env_id, seed):
    def thunk():
        try:
            env = gym.make(env_id, seed=seed)
        except TypeError:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def objective(config):

    obs, _ = envs.reset(seed=args.seed)
    actions = np.array(
        [
            config["params"][f"param_{x}"]
            for x in range(envs.single_action_space.shape[0])
        ]
    ).reshape(1, -1)
    episodic_return = None

    while episodic_return is None:
        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions
        )
        if "final_info" in infos:
            for info in infos["final_info"]:
                episodic_return = info["episode"]["r"]
                break

    train.report({"episodic_return": episodic_return[0]})


args = tyro.cli(Args)
date = time.strftime("%Y-%m-%d", time.gmtime(time.time()))

envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
config = {}
for x in range(envs.single_action_space.shape[0]):
    config[f"param_{x}"] = tune.uniform(
        envs.single_action_space.low[x], envs.single_action_space.high[x]
    )

search_space = config
search_alg = OptunaSearch(seed=args.seed)

ray_kwargs = {}
ray_kwargs["runtime_env"] = {"working_dir": BASE_DIR, "conda": "venv"}
try:
    if os.environ["ip_head"]:
        ray_kwargs["address"] = os.environ["ip_head"]
except Exception:
    ray_kwargs["num_cpus"] = 2

ray.init(**ray_kwargs)

trainable = tune.with_resources(objective, resources={"cpu": 1, "gpu": 0})

RESULTS_DIR = f"{BASE_DIR}/baselines/optuna/results/{args.exp_id}-optuna"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

storage_path = f"{RESULTS_DIR}/{args.algo}_run_{date}"

if tune.Tuner.can_restore(storage_path):
    print("Restoring old run ...")
    tuner = tune.Tuner.restore(
        storage_path, trainable=trainable, resume_errored=True
    )
else:
    print("Starting from scratch ...")
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="episodic_return",
            mode="max",
            search_alg=search_alg,
            num_samples=100,
            max_concurrent_trials=32,
        ),
        param_space={
            "scaling_config": train.ScalingConfig(use_gpu=False),
            "params": search_space,
        },
        run_config=train.RunConfig(
            storage_path=storage_path,
            name=f"pn341_ray_slurm_{args.exp_id}_{args.algo}",
            stop={"time_total_s": 2 * 60 * 60},
        ),
    )
results = tuner.fit()
best_result = results.get_best_result()
print("Best metrics:", best_result.metrics)

with open(f"{RESULTS_DIR}/best_{args.algo}_result_{date}.pkl", "wb") as file:
    pickle.dump(best_result.metrics, file)
