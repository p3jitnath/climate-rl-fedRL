import argparse
import math

from flwr_client import generate_client_fn

import flwr as fl


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

    # Define the client function
    client_fn = generate_client_fn(actor_layer_size=256)

    # Define the federated learning strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=0.0,
    )

    # Calculate total CPU and GPU requirements
    total_cpus = num_clients * 4  # Each client gets 4 CPUs
    total_gpus = math.ceil(num_clients * 0.25)  # Each client gets 0.25 GPU

    # Start the simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        strategy=strategy,
        ray_init_args={"num_cpus": total_cpus, "num_gpus": total_gpus},
        client_resources={"num_cpus": 4, "num_gpus": 0.25},
        config=fl.server.ServerConfig(num_rounds=10),
    )


if __name__ == "__main__":
    main()
