import os
import time

from smartsim import Experiment
from smartsim.status import SmartSimStatus

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-f2py/cm-v6"
# SCM_EXECUTABLE = "scm.o"
FLWR_SCRIPT = "flwr_main.py"
SEEDS = [x for x in range(16)]  # Add more seeds here if needed


def get_redis_port():
    """Retrieve the Redis port from the SSDB environment variable."""
    redis_port = os.getenv("SSDB")
    if not redis_port:
        raise EnvironmentError("The environment variable $SSDB is not set.")
    return int(redis_port.split(":")[-1])


def create_and_start_model(exp, name, exe_path, args, block=False):
    """Create and start a model with specified settings."""
    settings = exp.create_run_settings(exe=exe_path, exe_args=args)
    model = exp.create_model(name, run_settings=settings)
    exp.start(model, block=block)
    print(f"{name} started with args: {args}", flush=True)
    return model


def wait_for_completion(exp, models, label=""):
    """Wait for all specified models to complete."""
    print(f"Waiting for {label} processes to complete...", flush=True)
    while True:
        statuses = exp.get_status(*models)
        if all(
            status == SmartSimStatus.STATUS_COMPLETED for status in statuses
        ):
            print(f"All {label} processes completed successfully.", flush=True)
            break
        elif any(
            status
            in [SmartSimStatus.STATUS_FAILED, SmartSimStatus.STATUS_CANCELLED]
            for status in statuses
        ):
            print(
                f"One or more {label} processes failed or were cancelled.",
                flush=True,
            )
            break
        time.sleep(2)  # Wait before checking status again


def main():
    # Initialise SmartSim Experiment
    exp = Experiment("FLWR_Orchestrator", launcher="local")

    # Retrieve Redis port and start Redis database
    redis_port = get_redis_port()
    redis_model = exp.create_database(port=redis_port, interface="lo")
    exp.start(redis_model)
    print(f"Redis Database started on port {redis_port}.", flush=True)

    # Start SCM processes with different seeds
    # scm_models = []
    # for seed in SEEDS:
    #     model = create_and_start_model(
    #         exp,
    #         f"SCM_Seed_{seed}",
    #         f"{BASE_DIR}/{SCM_EXECUTABLE}",
    #         [str(seed)],
    #         block=False,
    #     )
    #     scm_models.append(model)

    # Start FLWR orchestrator
    flwr_model = create_and_start_model(
        exp,
        "FLWR_Orchestrator",
        "python",
        [f"{BASE_DIR}/{FLWR_SCRIPT}", "--num_clients", f"{len(SEEDS)}"],
        block=False,
    )

    # Wait for FLWR process to complete
    wait_for_completion(exp, [flwr_model], label="FLWR")

    # Stop all processes after completion
    exp.stop(redis_model)  # exp.stop(*scm_models, redis_model)
    print("Experiment completed successfully.", flush=True)


if __name__ == "__main__":
    main()
