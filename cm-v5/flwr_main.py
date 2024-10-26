import flwr as fl
from flwr_client import generate_client_fn


def main():
    client_fn = generate_client_fn(actor_layer_size=256)
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2, min_available_clients=2, fraction_evaluate=0.0
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        strategy=strategy,
        ray_init_args={"num_cpus": 4, "num_gpus": 1},
        client_resources={
            "num_cpus": 4,
            "num_gpus": 0.25,
        },
        config=fl.server.ServerConfig(num_rounds=12),
    )


if __name__ == "__main__":
    main()
