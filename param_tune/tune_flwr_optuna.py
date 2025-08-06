import fcntl
import json
import os
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import ray
import tyro
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
sys.path.append(BASE_DIR)

from param_tune.config import config


@dataclass
class Args:
    algo: str = "ddpg"
    """name of rl-algo"""
    exp_id: str = "ebm-v3-fedRL-L-20k-a2-fed05"
    """name of the experiment"""
    env_id: str = "EnergyBalanceModel-v3"
    """name of the environment"""
    actor_layer_size: Optional[int] = None
    """layer size for the actor network"""
    critic_layer_size: Optional[int] = None
    """layer size for the critic network"""
    opt_timesteps: int = 20000
    """timestep duration for one single optimisation run"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    flwr_client: int = 0
    """flwr client id for Federated Learning"""
    max_trials: int = 100
    """max number of optuna trials"""


def get_trial_idx():
    if not os.path.exists(args.trial_index_path):
        with open(args.trial_index_path, "w") as file:
            file.write("0")
    with open(args.trial_index_path, "r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        idx = int(file.read().strip())
        file.seek(0)
        file.write(str(idx + 1))
        file.truncate()
    return idx


def objective(config):
    study_id = random.randint(1000000000, 9999999999)
    trial_idx = get_trial_idx()
    tmp_file = f"{args.algo}_{study_id}.tmp"
    results_path = f"{BASE_DIR}/param_tune/tmp/{tmp_file}"

    params = {}
    # params["trial_idx"] = trial_idx
    params["env_id"] = args.env_id
    params["opt_timesteps"] = args.opt_timesteps
    params["num_steps"] = args.num_steps
    params["write_to_file"] = results_path

    for param in config["params"]:
        if param == "actor_critic_layer_size":
            actor_layer_size = critic_layer_size = config["params"][param]
            params["actor_layer_size"] = actor_layer_size
            params["critic_layer_size"] = critic_layer_size
        else:
            params[param] = config["params"][param]

    with open(
        f"{BASE_DIR}/param_tune/tmp/fedRL_{args.exp_id}_{args.algo}_{args.flwr_client}_{study_id}_T{trial_idx}.json",
        "w",
    ) as file:
        json.dump(params, file, indent=4)

    counter = 0
    while not os.path.exists(results_path):
        time.sleep(15)
        counter += 1
        if counter >= 1 * 60 * (60 // 15):
            raise RuntimeError("An error has occured.")

    with open(results_path, "rb") as f:
        results_dict = pickle.load(f)
        train.report(
            {"last_episodic_return": results_dict["last_episodic_return"]}
        )


args = tyro.cli(Args)
args.num_clients = int(re.search(r"a(\d+)", args.exp_id).group(1))
args.trial_index_path = f"{BASE_DIR}/param_tune/tmp/fedRL_{args.exp_id}_{args.algo}_{args.flwr_client}.gc"
date = time.strftime("%Y-%m-%d", time.gmtime(time.time()))

search_space = config[args.algo]
if args.actor_layer_size and args.critic_layer_size:
    search_space["actor_critic_layer_size"] = tune.choice(
        [args.actor_layer_size]
    )
search_alg = OptunaSearch(seed=42)

ray_kwargs = {}
ray_kwargs["runtime_env"] = {
    "working_dir": BASE_DIR,
    "conda": "venv",
    "excludes": [
        "runs/",
        "records/",
        "videos/",
        "results/",
        "wandb/",
        "notebooks/",
        "archive/" "slurm/",
        ".git/",
    ],
}
ray_kwargs["include_dashboard"] = False
ray_kwargs["num_cpus"] = 12

ray.init(**ray_kwargs)

trainable = tune.with_resources(
    objective, resources={"cpu": 1, "gpu": 0}
)  # resources={"cpu": 1, "gpu": 0.25}

RESULTS_DIR = f"{BASE_DIR}/param_tune/results/{args.exp_id}"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

storage_path = f"{RESULTS_DIR}/{args.algo}_run_{args.flwr_client}_{date}"

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
            metric="last_episodic_return",
            mode="max",
            search_alg=search_alg,
            num_samples=args.max_trials,
            max_concurrent_trials=12,
        ),
        param_space={
            "scaling_config": train.ScalingConfig(
                use_gpu=False
            ),  # use_gpu=True
            "params": search_space,
        },
        run_config=train.RunConfig(
            storage_path=storage_path,
            name=f"pn341_ray_slurm_{args.exp_id}_{args.algo}_{args.flwr_client}",
            stop={"time_total_s": 24 * 60 * 60},
        ),
    )

results = tuner.fit()
best_result = results.get_best_result()
print("Best metrics:", best_result.metrics)

with open(
    f"{RESULTS_DIR}/best_{args.algo}_{args.flwr_client}_result_{date}.pkl",
    "wb",
) as file:
    pickle.dump(best_result.metrics, file)
