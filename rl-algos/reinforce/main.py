import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import f2py_climate_envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from reinforce_actor import Actor
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
sys.path.append(BASE_DIR)

from fedrl.fedrl import FedRL
from param_tune.utils.no_op_summary_writer import NoOpSummaryWriter

os.environ["WANDB__SERVICE_WAIT"] = "600"
os.environ["MUJOCO_GL"] = "egl"
date = time.strftime("%Y-%m-%d", time.gmtime(time.time()))


@dataclass
class Args:
    exp_name: str = "reinforce_torch"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "personal-p3jitnath"
    """the entity (team) of wandb's project"""
    wandb_group: str = date
    """the group name under wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_video_freq: int = 100
    """episode frequency at which to capture video"""

    env_id: str = "SimpleClimateBiasCorrection-v0"
    """the environment id of the environment"""
    total_timesteps: int = 60000
    """total timesteps of the experiments"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    num_envs: int = 1
    """the number of sequential game environments"""

    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""

    optimise: bool = False
    """whether to modify output for hyperparameter optimisation"""
    write_to_file: str = ""
    """filename to write last episode return"""
    optim_group: str = ""
    """folder name under results to load optimised set of params"""
    opt_timesteps: Optional[int] = None
    """timestep duration for one single optimisation run"""

    actor_layer_size: int = 128
    """layer size for the actor network"""
    critic_layer_size: int = 128
    """layer size for the critic network"""

    flwr_client: Optional[int] = None
    """flwr client id for Federated Learning"""
    flwr_episodes: int = 5
    """the number of episodes after each flwr update"""
    flwr_actor: bool = True
    """whether to exchange actor network weights"""

    def __post_init__(self):
        if self.flwr_client is not None:
            self.track = False
            # self.capture_video = False

        if self.optimise:
            self.track = False
            self.capture_video = False
            self.total_timesteps = self.opt_timesteps

        if self.optim_group:
            algo = self.exp_name.split("_")[0]
            with open(
                f"{BASE_DIR}/param_tune/results/{self.optim_group}/best_results.json",
                "r",
            ) as file:
                opt_params = {
                    k: v
                    for k, v in json.load(file)[algo].items()
                    if k not in {"algo", "episodic_return", "date"}
                }
                for key, value in opt_params.items():
                    if key == "actor_critic_layer_size":
                        setattr(self, "actor_layer_size", value)
                        setattr(self, "critic_layer_size", value)
                    elif hasattr(self, key):
                        setattr(self, key, value)


class RecordSteps:
    def __init__(self, steps_folder, optimise):
        self.steps_folder = steps_folder
        self.optimise = optimise
        self._clear()

    def _clear(self):
        self.global_steps = []
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []

    def reset(self):
        self._clear()

    def add(self, global_step, obs, next_obs, actions, rewards):
        if not self.optimise:
            self.global_steps.append(global_step)
            self.obs.append(obs)
            self.next_obs.append(next_obs)
            self.actions.append(actions)
            self.rewards.append(rewards)

    def save(self, global_step, actor, episodic_return):
        if not self.optimise:
            torch.save(
                {
                    "global_steps": np.array(self.global_steps).squeeze(),
                    "obs": np.array(self.obs).squeeze(),
                    "next_obs": np.array(self.next_obs).squeeze(),
                    "actions": np.array(self.actions).squeeze(),
                    "rewards": np.array(self.rewards).squeeze(),
                    "actor": actor.state_dict(),
                    "episodic_return": episodic_return,
                },
                f"{self.steps_folder}/step_{global_step}.pth",
            )
            self.reset()


def make_env(
    env_id, seed, cid, idx, capture_video, run_name, capture_video_freq
):
    def thunk():
        if capture_video and idx == 0:
            try:
                env = gym.make(
                    env_id, seed=seed, cid=cid, render_mode="rgb_array"
                )
            except TypeError:
                env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"{BASE_DIR}/videos/{run_name}",
                episode_trigger=lambda x: (x == 0)
                or (
                    x % capture_video_freq == (capture_video_freq - 1)
                ),  # add 1 to the episode count generated by gym
            )
        else:
            try:
                env = gym.make(env_id, seed=seed, cid=cid)
            except TypeError:
                env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


args = tyro.cli(Args)

if args.flwr_client is not None:
    run_name = f"{args.wandb_group}/{args.env_id}__{args.exp_name}__{args.seed}__{args.flwr_client}__{int(time.time())}"
else:
    run_name = f"{args.wandb_group}/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

records_folder = f"{BASE_DIR}/records/{run_name}"
if not args.optimise:
    os.makedirs(records_folder, exist_ok=True)
rs = RecordSteps(records_folder, args.optimise)

if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        group=args.wandb_group,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )


if args.optimise:
    writer = NoOpSummaryWriter()
else:
    writer = SummaryWriter(f"{BASE_DIR}/runs/{run_name}")

writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device(
    "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
)
print(f"device: {device}", flush=True)
print(f"actor layer size: {args.actor_layer_size}", flush=True)

# 0. env setup
envs = gym.vector.SyncVectorEnv(
    [
        make_env(
            args.env_id,
            args.seed,
            args.flwr_client,
            0,
            args.capture_video,
            run_name,
            args.capture_video_freq,
        )
    ]
)

assert isinstance(
    envs.single_action_space, gym.spaces.Box
), "only continuous action space is supported"

actor = Actor(envs, args.actor_layer_size).to(device)
optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)
envs.single_observation_space.dtype = np.float32


# util function to calculate the discounted normalized returns
def compute_returns(rewards):
    rewards = np.array(rewards)
    returns = np.zeros_like(rewards, dtype=np.float32)
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + args.gamma * R
        returns[t] = R
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    return returns


# 1. start the game
global_step = 0
start_time = time.time()

# initialise for federated learning
if args.flwr_client is not None:

    weights_folder = f"{BASE_DIR}/records/{run_name}/fedrl-weights"
    os.makedirs(f"{weights_folder}/actor", exist_ok=True)

    fedRL = FedRL(
        args.seed,
        None,
        actor,
        None,
        args.flwr_client,
        None,
        args.flwr_actor,
        None,
        weights_folder,
    )

    # save the current initial actor weights when using federated learning
    # print('[RL Agent]', args.seed, "saving local actor weights", flush=True)
    fedRL.save_weights(0)

    # load the new actor weights from the global server
    # print('[RL Agent]', args.seed, "loading global actor weights", flush=True)
    fedRL.load_weights(0)

obs, _ = envs.reset(seed=args.seed)
args.num_episodes = args.total_timesteps // args.num_steps

for episode in range(1, args.num_episodes + 1):
    log_probs, rewards = [], []
    for step in range(1, args.num_steps + 1):
        # 2. retrieve action(s)
        global_step += args.num_envs
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        action, log_prob, _ = actor.get_action(obs)

        # 3. execute the game and log data
        next_obs, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )

        if "final_info" in infos:
            for info in infos["final_info"]:
                if args.flwr_client is not None:
                    print(
                        f"flwr_client={args.flwr_client}, seed={args.seed}, global_step={global_step}, episodic_return={info['episode']['r']}",
                        flush=True,
                    )
                else:
                    print(
                        f"seed={args.seed}, global_step={global_step}, episodic_return={info['episode']['r']}",
                        flush=True,
                    )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                if (
                    global_step % (args.num_steps * args.capture_video_freq)
                    == 0
                ):
                    rs.save(global_step, actor, info["episode"]["r"])
                break

        rs.add(
            global_step,
            obs.cpu().numpy(),
            next_obs,
            action.detach().cpu().numpy(),
            reward,
        )

        # 4. save data to the list of records
        log_probs.append(log_prob)
        rewards.append(reward)
        obs = next_obs

    # 5. compute the returns and the policy loss
    returns = compute_returns(rewards)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    actor_loss = -(torch.stack(log_probs) * returns).sum()

    optimizer.zero_grad()
    actor_loss.backward()
    optimizer.step()

    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
    writer.add_scalar(
        "charts/SPS",
        int(global_step / (time.time() - start_time)),
        global_step,
    )

    if args.flwr_client is not None and "final_info" in infos:
        if episode % args.flwr_episodes == 0:
            for info in infos["final_info"]:
                # save the current actor and/or critic weights when using federated learning
                # print('[RL Agent]', args.seed, "saving local weights", flush=True)
                fedRL.save_weights(global_step)

                # load the new actor and/or critic weights from the global server
                # print('[RL Agent]', args.seed, "loading global weights", flush=True)
                fedRL.load_weights(global_step)

                break

    if episode == args.num_episodes:
        if args.write_to_file:
            episodic_return = info["episode"]["r"][0]
            with open(args.write_to_file, "wb") as file:
                pickle.dump(
                    {
                        "num_episodes": args.num_episodes,
                        "last_episodic_return": episodic_return,
                    },
                    file,
                )

envs.close()
writer.close()
