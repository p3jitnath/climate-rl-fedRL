import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import tyro

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
sys.path.append(BASE_DIR)


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
    max_trials: int = 100
    """max number of optuna trials"""
    flwr_episodes: int = 5
    """the number of episodes after each flwr update"""


args = tyro.cli(Args)
args.num_clients = int(re.search(r"a(\d+)", args.exp_id).group(1))

if args.actor_layer_size is None or args.critic_layer_size is None:
    # with open(
    #     f"{BASE_DIR}/param_tune/results/ebm-v1-optim-L-20k/best_results.json",
    #     "r",
    # ) as file:
    #     opt_params = {
    #         k: v
    #         for k, v in json.load(file)[args.algo].items()
    #         if k not in {"algo", "episodic_return", "date"}
    #     }
    #     for key, value in opt_params.items():
    #         if key == "actor_critic_layer_size":
    #             args.actor_layer_size = value
    #             args.critic_layer_size = value
    args.actor_layer_size = args.critic_layer_size = 256

processes = []
for cid in range(args.num_clients):
    cmd = (
        f"python -u {BASE_DIR}/param_tune/tune_flwr_optuna.py "
        f"--algo {args.algo} "
        f"--exp_id {args.exp_id} "
        f"--env_id {args.env_id} "
        f"--actor_layer_size {args.actor_layer_size} "
        f"--critic_layer_size {args.critic_layer_size} "
        f"--opt_timesteps {args.opt_timesteps} "
        f"--num_steps {args.num_steps} "
        f"--flwr_client {cid} "
        f"--max_trials {args.max_trials} "
    )
    print(cmd, flush=True)
    process = subprocess.Popen(
        cmd.split(), stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setsid
    )
    processes.append((cid, process))


def terminate_all(signum, frame):
    for cid, process in processes:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    sys.exit(0)


signal.signal(signal.SIGINT, terminate_all)
TMP_DIR = f"{BASE_DIR}/param_tune/tmp/"
seen = set()


def parse_file_name(fname):
    match = re.search(
        rf"{re.escape(args.algo)}_(\d+)_.*?_T(\d+)\.json$", fname
    )
    if match:
        cid, tidx = match.groups()
        return int(cid), int(tidx)
    return None


while True:
    grouped = defaultdict(set)
    pattern = os.path.join(TMP_DIR, f"fedRL_{args.exp_id}_{args.algo}_*")
    files = glob.glob(pattern)

    for path in files:
        fname = os.path.basename(path)
        parsed = parse_file_name(fname)
        if parsed is None:
            continue

        cid, tidx = parsed
        if tidx in seen:
            continue

        grouped[tidx].add(cid)

    for tidx in sorted(grouped.keys()):
        if tidx in seen:
            continue

        if len(grouped[tidx]) == args.num_clients:
            cmd = (
                "sbatch "
                f"{BASE_DIR}/flwr_smartsim.sh "
                f"--rl_algo {args.algo} "
                f"--env_id {args.env_id} "
                f"--tag {args.exp_id} "
                f"--flwr_episodes {str(args.flwr_episodes)} "
                f"--num_clients {str(args.num_clients)} "
                f"--opt_timesteps {str(args.opt_timesteps)} "
                "--seed 1 "
                "--optimise true "
                f"--opt_trial_idx {str(tidx)} "
            )
            print(cmd, flush=True)
            subprocess.Popen(cmd.split())
            seen.add(tidx)

    if len(seen) == args.max_trials:
        break

    time.sleep(2)

for cid, process in processes:
    try:
        os.waitpid(process.pid, 0)
    except ChildProcessError:
        pass
