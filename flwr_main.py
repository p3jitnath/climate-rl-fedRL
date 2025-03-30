import argparse
import math
import os
import pickle

import flwr as fl
import numpy as np
import smartredis
from flwr.common import FitIns

from flwr_client import generate_client_fn


class FedAvgWithBuffer(fl.server.strategy.FedAvg):
    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        fraction_evaluate=0.0,
        **kwargs,
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )
        self.global_rb = []

        # SmartRedis setup
        self.REDIS_ADDRESS = os.getenv("SSDB")
        if self.REDIS_ADDRESS is None:
            raise EnvironmentError("SSDB environment variable is not set.")
        self.redis = smartredis.Client(
            address=self.REDIS_ADDRESS, cluster=False
        )
        print(
            f"[Flwr Main] Connected to Redis server: {self.REDIS_ADDRESS}",
            flush=True,
        )

    def configure_fit(self, server_round, parameters, client_manager):
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        fit_configurations = []
        for client in clients:
            fit_configurations.append(
                (client, FitIns(parameters, {"server_round": server_round}))
            )
        return fit_configurations

    def aggregate_fit(self, server_round, results, failures):
        # Add the new replay buffer entries to a global buffer
        # print("[Flwr Main] - performing aggregation", flush=True)
        for _, fit_res in results:
            new_rb_entries = fit_res.metrics.get(
                "new_replay_buffer_entries", None
            )
            if new_rb_entries is not None:
                new_rb_entries = pickle.loads(new_rb_entries)
                self.global_rb.extend(new_rb_entries)

        self.redis.put_tensor(
            "replay_buffer_global",
            np.frombuffer(pickle.dumps(self.global_rb), dtype=np.uint8),
        )

        return super().aggregate_fit(server_round, results, failures)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Flower Simulation with N clients."
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        required=True,
        help="Number of simulated clients (N).",
    )
    args = parser.parse_args()
    num_clients = args.num_clients
    is_distributed = bool(int(os.environ.get("DISTRIBUTED", 0)))

    # Define the client function
    client_fn = generate_client_fn(
        actor_critic_layer_size=256, is_distributed=is_distributed
    )

    # Define the federated learning strategy
    strategy = FedAvgWithBuffer(
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=0.0,
    )

    # Calculate total CPU and GPU requirements only in the non-distributed case
    ray_init_args = None
    if not is_distributed:
        total_cpus = num_clients * 3 + 1  # Each client gets 3 CPUs + 1 extra
        total_gpus = 0  # math.ceil(num_clients * 0.25)
        ray_init_args = {
            "num_cpus": total_cpus,
            "num_gpus": total_gpus,
        }

    # Start the simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        strategy=strategy,
        ray_init_args=ray_init_args,
        client_resources={"num_cpus": 3, "num_gpus": 0},
        config=fl.server.ServerConfig(
            num_rounds=6
        ),  # +1 to have 1 extra round # steps = num_rounds * 200 * flwr_episodes
    )


if __name__ == "__main__":
    main()
