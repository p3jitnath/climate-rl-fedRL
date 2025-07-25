import os
import time

import psutil
from smartsim import Experiment
from smartsim.status import SmartSimStatus

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
ENVIRONMENT_DIR = f"{BASE_DIR}/fedrl-climate-envs/fedrl_climate_envs/envs"
PYTHON_EXE = "/home/users/p341cam/miniconda3/envs/venv/bin/python"

FLWR_EXE = "flwr/flwr_main.py"
SCM_EXE = "scm.o"
CLIMLAB_EXE = "climlab_ebm.py"

RL_ALGO = os.getenv("RL_ALGO")
WANDB_GROUP = os.getenv("WANDB_GROUP")
ENV_ID = os.getenv("ENV_ID")
SEED = os.getenv("SEED")

NUM_CLIENTS = int(os.getenv("NUM_CLIENTS"))
CLIENTS = [x for x in range(NUM_CLIENTS)]  # Add more clients here if needed

# SBATCH_ARGS = {
#     "nodes": 1,
#     "ntasks-per-node": 1,
#     "cpus-per-task": 2,
#     "mem-per-cpu": "8G",
#     "time": "01:00:00",
#     "partition": "test",
# }


# FLWR_SBATCH_ARGS = {
#     "nodes": 1,
#     "ntasks-per-node": 1,
#     "cpus-per-task": NUM_CLIENTS + 1,
#     "mem-per-cpu": "8G",
#     "time": "01:00:00",
#     "partition": "test",
# }

# NUM_CPUs : 1 (THIS) + 1 (CLIMATE MODEL) + NUM_CLIENTS+1 (FLWR)


def get_redis_port():
    """Retrieve the Redis port from the SSDB environment variable."""
    redis_port = os.getenv("SSDB")
    if not redis_port:
        raise EnvironmentError("The environment variable $SSDB is not set.")
    return int(redis_port.split(":")[-1])


def create_and_start_model(
    exp, name, exe_path, args, block=False, batch_settings=None
):
    """Create and start a model with specified settings."""
    run_settings = exp.create_run_settings(exe=exe_path, exe_args=args)
    model = exp.create_model(
        name, run_settings=run_settings, batch_settings=batch_settings
    )
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
    exp_name = f"SM-FLWR_Orchestrator_{RL_ALGO}_{WANDB_GROUP}_{SEED}"
    exp_dir = f"{BASE_DIR}/SM-FLWR/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    exp = Experiment(exp_name, exp_path=exp_dir, launcher="local")

    # Retrieve Redis port and start Redis database
    interfaces = list(psutil.net_if_addrs().keys())
    print("Available network interfaces:", interfaces, flush=True)
    redis_model = exp.create_database(
        port=get_redis_port(), interface=interfaces[1]
    )
    print(
        f"Running Redis database on {os.getenv('SSDB')} via {interfaces[1]}",
        flush=True,
    )
    exp.start(redis_model)

    # SBATCH_ARGS["export"] = FLWR_SBATCH_ARGS["export"] = f"SSDB={redis_port}"

    # Generate batch settings
    # batch_settings = exp.create_batch_settings(batch_args=SBATCH_ARGS)
    # flwr_batch_settings = exp.create_batch_settings(
    #     batch_args=FLWR_SBATCH_ARGS
    # )

    # if ENV_ID in ["SimpleClimateBiasCorrection-v0"]:
    #     # Start SCM processes with different seeds
    #     scm_models = []
    #     for cid in CLIENTS:
    #         model = create_and_start_model(
    #             exp,
    #             f"SCM_{cid}",
    #             f"{ENVIRONMENT_DIR}/{SCM_EXE}",
    #             [str(cid)],
    #             block=False,
    #             # batch_settings=batch_settings
    #         )
    #         scm_models.append(model)

    if ENV_ID in ["EnergyBalanceModel-v3"]:
        ebm_model = create_and_start_model(
            exp,
            "EBM",
            PYTHON_EXE,
            [
                f"{ENVIRONMENT_DIR}/{CLIMLAB_EXE}",
                "--num_clients",
                f"{len(CLIENTS)}",
            ],
            block=False,
        )

    # Start FLWR orchestrator
    flwr_model = create_and_start_model(
        exp,
        "FLWR_Orchestrator",
        PYTHON_EXE,
        [f"{BASE_DIR}/{FLWR_EXE}", "--num_clients", f"{len(CLIENTS)}"],
        block=False,
    )

    # Wait for FLWR process to complete
    wait_for_completion(exp, [flwr_model], label="FLWR")

    # Stop all processes after completion
    # if ENV_ID in ["SimpleClimateBiasCorrection-v0"]:
    #     exp.stop(*scm_models)
    if ENV_ID in ["EnergyBalanceModel-v3"]:
        exp.stop(ebm_model)
    exp.stop(redis_model)
    print("Experiment completed successfully.", flush=True)


if __name__ == "__main__":
    main()
