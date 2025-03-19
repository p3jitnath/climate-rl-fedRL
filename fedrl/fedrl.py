import os
import pickle

import numpy as np
import torch
from smartredis import Client


class FedRL:
    def __init__(
        self,
        seed,
        actor,
        critic,
        flwr_client,
        flwr_actor,
        flwr_critic,
        weights_folder,
    ):
        self.seed = seed
        self.actor = actor
        self.critic = critic
        self.flwr_client = flwr_client

        self.new_rb_entries = []
        self.global_rb = None
        self.flwr_actor = flwr_actor
        self.flwr_critic = flwr_critic
        self.weights_folder = weights_folder

        # SmartRedis setup
        self.REDIS_ADDRESS = os.getenv("SSDB")
        if self.REDIS_ADDRESS is None:
            raise EnvironmentError("SSDB environment variable is not set.")
        self.redis = Client(address=self.REDIS_ADDRESS, cluster=False)
        print(
            f"[RL Agent] Connected to Redis server: {self.REDIS_ADDRESS}",
            flush=True,
        )

    # load the latest weights from Redis
    def load_weights(self, step_count):
        # 1. Actor
        if self.flwr_actor:
            # Wait for signal that weights are available
            while not self.redis.tensor_exists(
                f"actor_network_weights_g2c_s{self.seed}"
            ):
                pass

            # Retrieve and reshape weights tensor based on Actor's structure
            actor_weights = self.redis.get_tensor(
                f"actor_network_weights_g2c_s{self.seed}"
            )
            # print('[RL Agent] Actor', self.seed, 'L', actor_weights[0:5], flush=True)

            old_actor_params = [
                param.clone().detach() for param in self.actor.parameters()
            ]

            offset = 0
            for param in self.actor.parameters():
                size = np.prod(param.shape)
                param.data.copy_(
                    torch.tensor(
                        actor_weights[offset : offset + size].reshape(
                            param.shape
                        )
                    )
                )
                offset += size

            if self.weights_folder:
                torch.save(
                    self.actor.state_dict(),
                    f"{self.weights_folder}/actor/actor-fedRL-{step_count}.pt",
                )

            actor_diff_norm = sum(
                torch.norm(old - new)
                for old, new in zip(old_actor_params, self.actor.parameters())
            )
            print(f"[RL Agent] Actor norm: {actor_diff_norm}")

            # Clear weights and signal to reset for the next round
            self.redis.delete_tensor(f"actor_network_weights_g2c_s{self.seed}")

        # 2. Critic
        if self.flwr_critic:
            # Wait for signal that weights are available
            while not self.redis.tensor_exists(
                f"critic_network_weights_g2c_s{self.seed}"
            ):
                pass

            # Retrieve and reshape weights tensor based on Critic's structure
            critic_weights = self.redis.get_tensor(
                f"critic_network_weights_g2c_s{self.seed}"
            )
            # print('[RL Agent] Critic', self.seed, 'L', critic_weights[0:5], flush=True)

            old_critic_params = [
                param.clone().detach() for param in self.critic.parameters()
            ]

            offset = 0
            for param in self.critic.parameters():
                size = np.prod(param.shape)
                param.data.copy_(
                    torch.tensor(
                        critic_weights[offset : offset + size].reshape(
                            param.shape
                        )
                    )
                )
                offset += size

            if self.weights_folder:
                torch.save(
                    self.actor.state_dict(),
                    f"{self.weights_folder}/critic/critic-fedRL-{step_count}.pt",
                )

            critic_diff_norm = sum(
                torch.norm(old - new)
                for old, new in zip(
                    old_critic_params, self.critic.parameters()
                )
            )
            print(f"[RL Agent] Critic norm: {critic_diff_norm}")

            # Clear weights and signal to reset for the next round
            self.redis.delete_tensor(
                f"critic_network_weights_g2c_s{self.seed}"
            )

    # save updated weights to Redis
    def save_weights(self, step_count):
        # 1. Actor
        actor_weights = np.concatenate(
            [
                param.data.cpu().numpy().flatten()
                for param in self.actor.parameters()
            ]
        )
        self.redis.put_tensor(
            f"actor_network_weights_c2g_s{self.seed}", actor_weights
        )
        if self.weights_folder:
            torch.save(
                self.actor.state_dict(),
                f"{self.weights_folder}/actor/actor-{step_count}.pt",
            )
        # print('[RL Agent] Actor', self.seed, 'S', weights[0:5], flush=True)

        # 2. Critic
        critic_weights = np.concatenate(
            [
                param.data.cpu().numpy().flatten()
                for param in self.critic.parameters()
            ]
        )
        self.redis.put_tensor(
            f"critic_network_weights_c2g_s{self.seed}", critic_weights
        )
        if self.weights_folder:
            torch.save(
                self.actor.state_dict(),
                f"{self.weights_folder}/critic/critic-{step_count}.pt",
            )
        # print('[RL Agent] Critic', self.seed, 'S', weights[0:5], flush=True)

    # load the current global replay buffer from Redis
    def load_replay_buffer(self):
        # Wait for signal that aggregatd replay buffer is available
        while not self.redis.tensor_exists("replay_buffer_global"):
            pass

        # Retrieve the global replay buffer tensor
        self.global_rb = self.redis.get_tensor("replay_buffer_global")
        self.global_rb = pickle.loads(self.global_rb.tobytes())

    # save the new replay buffer entries to Redis
    def save_new_replay_buffer_entries(self):
        self.redis.put_tensor(
            f"new_replay_buffer_entries_c2g_s{self.seed}",
            np.frombuffer(pickle.dumps(self.new_rb_entries), dtype=np.uint8),
        )
        self.new_rb_entries = []
