import os
import pickle

import numpy as np
import torch
from smartredis import Client


class FedRL:
    def __init__(
        self,
        seed,
        agent,
        actor,
        critics,
        flwr_client,
        flwr_agent,
        flwr_actor,
        flwr_critics,
        weights_folder,
    ):
        self.seed = seed
        self.agent = agent
        self.actor = actor
        self.critics = critics
        self.flwr_client = flwr_client

        self.new_rb_entries = []
        self.global_rb = None
        self.flwr_agent = flwr_agent
        self.flwr_actor = flwr_actor
        self.flwr_critics = flwr_critics
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
        # 0. Agent
        if self.agent and self.flwr_agent:
            # Wait for signal that weights are available
            while not self.redis.tensor_exists(
                f"agent_network_weights_g2c_s{self.seed}"
            ):
                pass

            # Retrieve and reshape weights tensor based on Agent's structure
            agent_weights = self.redis.get_tensor(
                f"agent_network_weights_g2c_s{self.seed}"
            )
            # print('[RL Agent] Agent', self.seed, 'L', agent_weights[0:5], flush=True)

            old_agent_params = [
                param.clone().detach() for param in self.agent.parameters()
            ]

            offset = 0
            for param in self.agent.parameters():
                size = np.prod(param.shape)
                param.data.copy_(
                    torch.tensor(
                        agent_weights[offset : offset + size].reshape(
                            param.shape
                        )
                    )
                )
                offset += size

            if self.weights_folder:
                torch.save(
                    self.agent.state_dict(),
                    f"{self.weights_folder}/agent/agent-fedRL-{step_count}.pt",
                )

            agent_diff_norm = sum(
                torch.norm(old - new)
                for old, new in zip(old_agent_params, self.agent.parameters())
            )
            print(f"[RL Agent] Agent norm: {agent_diff_norm}")

            # Clear weights and signal to reset for the next round
            self.redis.delete_tensor(f"agent_network_weights_g2c_s{self.seed}")

        # 1. Actor
        if self.actor and self.flwr_actor:
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
        if self.critics and self.flwr_critics:
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
                param.clone().detach()
                for critic in self.critics
                for param in critic.parameters()
            ]

            offset = 0
            for critic in self.critics:
                for param in critic.parameters():
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
                for idx, critic in enumerate(self.critics):
                    torch.save(
                        critic.state_dict(),
                        f"{self.weights_folder}/critic/critic-{idx}-fedRL-{step_count}.pt",
                    )

            new_critic_params = [
                param.clone().detach()
                for critic in self.critics
                for param in critic.parameters()
            ]

            critic_diff_norm = sum(
                torch.norm(old - new)
                for old, new in zip(old_critic_params, new_critic_params)
            )
            print(f"[RL Agent] Critics norm: {critic_diff_norm}")

            # Clear weights and signal to reset for the next round
            self.redis.delete_tensor(
                f"critic_network_weights_g2c_s{self.seed}"
            )

    # save updated weights to Redis
    def save_weights(self, step_count):
        # 0. Agent
        if self.agent:
            agent_weights = np.concatenate(
                [
                    param.data.cpu().numpy().flatten()
                    for param in self.agent.parameters()
                ]
            )
            self.redis.put_tensor(
                f"agent_network_weights_c2g_s{self.seed}", agent_weights
            )
            if self.weights_folder:
                torch.save(
                    self.agent.state_dict(),
                    f"{self.weights_folder}/agent/agent-{step_count}.pt",
                )
            # print('[RL Agent] Agent', self.seed, 'S', weights[0:5], flush=True)

        # 1. Actor
        if self.actor:
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
            # print('[RL Agent] Actor', self.seed, 'S', actor_weights[0:5], flush=True)

        # 2. Critic
        if self.critics:
            critic_weights = np.concatenate(
                [
                    param.data.cpu().numpy().flatten()
                    for critic in self.critics
                    for param in critic.parameters()
                ]
            )
            self.redis.put_tensor(
                f"critic_network_weights_c2g_s{self.seed}", critic_weights
            )
            if self.weights_folder:
                for idx, critic in enumerate(self.critics):
                    torch.save(
                        critic.state_dict(),
                        f"{self.weights_folder}/critic/critic-{idx}-{step_count}.pt",
                    )
            # print('[RL Agent] Critic', self.seed, 'S', critic_weights[0:5], flush=True)

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
