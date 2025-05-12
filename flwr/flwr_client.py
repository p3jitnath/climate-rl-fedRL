import json
import os
import sys
from distutils.util import strtobool

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
EPISODE_LENGTH = 200

RL_ALGO = os.getenv("RL_ALGO")
ENV_ID = os.getenv("ENV_ID")
WANDB_GROUP = os.getenv("WANDB_GROUP")
OPTIM_GROUP = os.getenv("OPTIM_GROUP", None)
FLWR_ACTOR = bool(strtobool(os.getenv("FLWR_ACTOR")))
FLWR_CRITICS = bool(strtobool(os.getenv("FLWR_CRITICS")))

CRITIC_ALGOS = [
    ("avg", 1),
    ("dpg", 1),
    ("ddpg", 1),
    ("trpo", 1),
    ("sac", 2),
    ("td3", 2),
    ("tqc", 1),
]
PYTHON_EXE = "/home/users/p341cam/miniconda3/envs/venv/bin/python"
TQC_N_QUANTILES, TQC_N_CRITICS = 25, 5

sys.path.append(f"{BASE_DIR}/rl-algos/{RL_ALGO}")


import importlib
import pickle
import subprocess

import fedrl_climate_envs
import gymnasium as gym
import numpy as np
import smartredis

import flwr as fl

Agent = Actor = Critic = None
if RL_ALGO == "ppo":
    Agent = getattr(importlib.import_module(f"{RL_ALGO}_agent"), "Agent")
else:
    Actor = getattr(importlib.import_module(f"{RL_ALGO}_actor"), "Actor")

for algo, count in CRITIC_ALGOS:
    if RL_ALGO == algo:
        ncritics = count
        if RL_ALGO == "tqc":
            Critic = QCritic = getattr(
                importlib.import_module(f"{RL_ALGO}_quantile_critic"),
                "QuantileCritics",
            )
        else:
            Critic = getattr(
                importlib.import_module(f"{RL_ALGO}_critic"), "Critic"
            )


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, is_distributed):
        super().__init__()

        self.cid = cid
        self.seed = int(cid)
        self.is_distributed = is_distributed

        # SmartRedis setup
        self.REDIS_ADDRESS = os.getenv("SSDB")
        if self.REDIS_ADDRESS is None:
            raise EnvironmentError("SSDB environment variable is not set.")
        self.redis = smartredis.Client(
            address=self.REDIS_ADDRESS, cluster=False
        )
        print(
            f"[Flwr Client] Connected to Redis server: {self.REDIS_ADDRESS}",
            flush=True,
        )

        if OPTIM_GROUP:
            with open(
                f"{BASE_DIR}/param_tune/results/{OPTIM_GROUP}/best_results.json",
                "r",
            ) as file:
                opt_params = {
                    k: v
                    for k, v in json.load(file)[RL_ALGO].items()
                    if k not in {"algo", "episodic_return", "date"}
                }
                for key, value in opt_params.items():
                    if key == "actor_critic_layer_size":
                        actor_critic_layer_size = value
                        break
        else:
            actor_critic_layer_size = 256

        self.actor_layer_size = self.critic_layer_size = (
            actor_critic_layer_size
        )

        cmd = (
            f"""{PYTHON_EXE} -u {BASE_DIR}/rl-algos/{RL_ALGO}/main.py""" + " "
        )
        cmd += f"""--env_id {ENV_ID} --wandb_group {WANDB_GROUP} --num_steps {EPISODE_LENGTH} """
        cmd += f"--flwr_client {self.cid} --seed {self.seed}" + " "
        cmd += f"--actor_layer_size {self.actor_layer_size}" + " "
        cmd += (
            f"--critic_layer_size {self.critic_layer_size} --capture_video_freq 50"
            + " "
        )

        for name, enabled in (
            ("flwr_actor", FLWR_ACTOR),
            ("flwr_critics", FLWR_CRITICS),
        ):
            prefix = "" if enabled else "no-"
            cmd += f"--{prefix}{name}" + " "

        if not self.is_distributed:
            # Check if the command is already running using `pgrep`
            check_cmd = f"pgrep -f '{cmd}'"
            is_alive = subprocess.run(
                check_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # If no process is found, `pgrep` returns a non-zero exit code, so we start the process
            if is_alive.returncode != 0:
                print(cmd, flush=True)
                subprocess.Popen(cmd.split())
        else:
            is_alive = self.redis.tensor_exists(f"SIGALIVE_S{self.seed}")
            print("is_alive:", is_alive, flush=True)
            if not is_alive:
                print(cmd, flush=True)
                subprocess.Popen(cmd.split())

        def make_env(env_id, seed):
            def thunk():
                try:
                    env = gym.make(env_id, seed=seed)
                except TypeError:
                    env = gym.make(env_id)
                return env

            return thunk

        # env setup
        self.envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, self.seed)])

        assert isinstance(
            self.envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        self.agent = self.actor = self.critic = None

        if Agent:
            self.agent = Agent(
                self.envs, self.actor_layer_size, self.critic_layer_size
            )
        else:
            self.actor = Actor(self.envs, self.actor_layer_size)
            self.actor_offset = len(list(self.actor.parameters()))
            if Critic:
                if RL_ALGO == "tqc":
                    self.critics = [
                        QCritic(
                            self.envs,
                            TQC_N_QUANTILES,
                            TQC_N_CRITICS,
                            self.critic_layer_size,
                        )
                        for x in range(ncritics)
                    ]
                else:
                    self.critics = [
                        Critic(self.envs, self.critic_layer_size)
                        for x in range(ncritics)
                    ]

    def set_parameters(self, parameters):
        # 0. Agent
        # Flatten and stack all layer weights into one large tensor
        if self.agent:
            agent_weights = np.concatenate(
                [param.flatten() for param in parameters]
            )

            # Store weights in Redis and send a signal to indicate update
            self.redis.put_tensor(
                f"agent_network_weights_g2c_s{self.seed}", agent_weights
            )

        # 1. Actor
        # Flatten and stack all layer weights into one large tensor
        if self.actor:
            actor_weights = np.concatenate(
                [param.flatten() for param in parameters[: self.actor_offset]]
            )

            # Store weights in Redis and send a signal to indicate update
            self.redis.put_tensor(
                f"actor_network_weights_g2c_s{self.seed}", actor_weights
            )

        # 2. Critic
        # Flatten and stack all layer weights into one large tensor
        if self.critics:
            critic_weights = np.concatenate(
                [param.flatten() for param in parameters[self.actor_offset :]]
            )

            # Store weights in Redis and send a signal to indicate update
            self.redis.put_tensor(
                f"critic_network_weights_g2c_s{self.seed}", critic_weights
            )

    def get_parameters(self, config):
        # 0. Agent
        # Wait for signal that actor network weights are available
        if self.agent:
            while not self.redis.tensor_exists(
                f"agent_network_weights_c2g_s{self.seed}"
            ):
                continue

            # Retrieve and reshape weights tensor based on Agent's structure
            agent_weights = self.redis.get_tensor(
                f"agent_network_weights_c2g_s{self.seed}"
            )
            parameters = []
            offset = 0
            for param in self.agent.parameters():
                size = np.prod(param.shape)
                layer_weights = agent_weights[offset : offset + size].reshape(
                    param.shape
                )
                parameters.append(layer_weights)
                offset += size

            # Clear weights and signal to reset for the next round
            self.redis.delete_tensor(f"agent_network_weights_c2g_s{self.seed}")

        # 1. Actor
        # Wait for signal that actor network weights are available
        if self.actor:
            while not self.redis.tensor_exists(
                f"actor_network_weights_c2g_s{self.seed}"
            ):
                continue

            # Retrieve and reshape weights tensor based on Actor's structure
            actor_weights = self.redis.get_tensor(
                f"actor_network_weights_c2g_s{self.seed}"
            )
            parameters = []
            offset = 0
            for param in self.actor.parameters():
                size = np.prod(param.shape)
                layer_weights = actor_weights[offset : offset + size].reshape(
                    param.shape
                )
                parameters.append(layer_weights)
                offset += size

            # Clear weights and signal to reset for the next round
            self.redis.delete_tensor(f"actor_network_weights_c2g_s{self.seed}")

        # 2. Critic
        # Wait for signal that critic network weights are available
        if self.critics:
            while not self.redis.tensor_exists(
                f"critic_network_weights_c2g_s{self.seed}"
            ):
                continue

            # Retrieve and reshape weights tensor based on Critic's structure
            critic_weights = self.redis.get_tensor(
                f"critic_network_weights_c2g_s{self.seed}"
            )

            offset = 0
            for critic in self.critics:
                for param in critic.parameters():
                    size = np.prod(param.shape)
                    layer_weights = critic_weights[
                        offset : offset + size
                    ].reshape(param.shape)
                    parameters.append(layer_weights)
                    offset += size

            # Clear weights and signal to reset for the next round
            self.redis.delete_tensor(
                f"critic_network_weights_c2g_s{self.seed}"
            )

        return parameters

    def get_new_replay_buffer_entries(self):
        # Wait for signal that new replay buffer entries are available
        while not self.redis.tensor_exists(
            f"new_replay_buffer_entries_c2g_s{self.seed}"
        ):
            continue

        # Retrieve the new replay buffer entries
        new_rb_entries = self.redis.get_tensor(
            f"new_replay_buffer_entries_c2g_s{self.seed}"
        )
        new_rb_entries = pickle.loads(new_rb_entries.tobytes())

        return new_rb_entries

    def fit(self, parameters, config):
        # Update the actor network parameters
        # print(config['server_round'], self.seed, "[Flwr Client] - setting parameters", flush=True)
        self.set_parameters(parameters)

        # Retrieve updated parameters from Redis after RL processing
        # print(config['server_round'], self.seed, "[Flwr Client] - loading parameters", flush=True)
        updated_parameters = self.get_parameters(config)

        # Retrieve the new replay buffer entries from Redis
        # print(config['server_round'], self.seed, "[Flwr Client] - loading new replay buffer entries", flush=True)
        # new_rb_entries = (
        #     self.get_new_replay_buffer_entries()
        #     if RL_ALGO not in ["avg", "ppo", "trpo", "dpg"]
        #     else {}
        # )
        new_rb_entries = []

        return (
            updated_parameters,
            EPISODE_LENGTH,
            {"new_replay_buffer_entries": pickle.dumps(new_rb_entries)},
        )


def generate_client_fn(is_distributed=False):
    def client_fn(context):
        return FlowerClient(
            int(context.node_config["partition-id"]),
            is_distributed,
        ).to_client()

    return client_fn
