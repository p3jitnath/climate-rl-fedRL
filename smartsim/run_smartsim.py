import os
import random
import socket
import struct
import time
from distutils.util import strtobool

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
OPTIM_GROUP = os.getenv("OPTIM_GROUP")
SEED = os.getenv("SEED")
INFERENCE = bool(int(os.environ.get("INFERENCE", 0)))
GLOBAL = bool(int(os.environ.get("GLOBAL", 0)))

OPTIMISE = bool(strtobool(os.getenv("OPTIMISE")))
EXP_ID = os.getenv("TAG")
OPT_TRIAL_IDX = int(os.getenv("OPT_TRIAL_IDX"))

NUM_CLIENTS = int(os.getenv("NUM_CLIENTS"))


def get_ip_address():
    try:
        # Use a dummy connection to an external host
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Doesn't send data
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"  # fallback to localhost


def get_urandom_redis_port():
    """Generate a random Redis port using a hardware random number generator"""
    with open("/dev/urandom", "rb") as f:
        hrand_bytes = f.read(4)
    hrand = struct.unpack("I", hrand_bytes)[0]
    redis_port = random.randint(12581, 24580)
    redis_port += hrand % 1000
    return redis_port


def get_ssdb_redis_port():
    """Retrieve the Redis port from the SSDB environment variable."""
    redis_port = os.getenv("SSDB")
    if not redis_port:
        raise EnvironmentError("The environment variable $SSDB is not set.")
    return int(redis_port.split(":")[-1])


def create_and_start_model(
    exp, name, exe_path, args, block=False, batch_settings=None, colocate=False
):
    """Create and start a model with specified settings."""
    run_settings = exp.create_run_settings(
        run_command="" if colocate else "auto",
        exe=exe_path,
        exe_args=args,
    )

    if INFERENCE and not colocate:
        run_settings.set_nodes(1)
        run_settings.set_tasks_per_node(1)
        if RL_ALGO == "tqc":
            run_settings.set("gpus", "1")

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
    if OPTIMISE:
        exp_name = (
            f"SM-FLWR_Orchestrator_{RL_ALGO}_optuna_{EXP_ID}_{OPT_TRIAL_IDX}"
        )
    else:
        exp_name = f"SM-FLWR_Orchestrator_{RL_ALGO}_{WANDB_GROUP}_{SEED}"
    exp_dir = f"{BASE_DIR}/SM-FLWR/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    exp = Experiment(
        exp_name, exp_path=exp_dir, launcher="slurm" if INFERENCE else "local"
    )

    # print("[SmartSim MAIN]: ip address", get_ip_address(), flush=True)
    # print("[SmartSim MAIN]: hostname", socket.gethostname(), flush=True)

    # Retrieve Redis port and start Redis database
    interfaces = list(psutil.net_if_addrs().keys())
    print("Available network interfaces:", interfaces, flush=True)

    redis_db = exp.create_database(
        port=get_urandom_redis_port() if INFERENCE else get_ssdb_redis_port(),
        interface=interfaces[1],
    )
    if RL_ALGO == "tqc":
        redis_db.set_run_arg("gpus", "0")
    # print(exp.preview(redis_db, verbosity_level="developer"), flush=True)
    exp.start(redis_db)
    print(
        f"Running Redis database on {redis_db.get_address()[0]} via {interfaces[1]}",
        flush=True,
    )

    if ENV_ID in ["EnergyBalanceModel-v3"]:
        ebm_model = create_and_start_model(
            exp,
            "EBM",
            PYTHON_EXE,
            [
                f"{ENVIRONMENT_DIR}/{CLIMLAB_EXE}",
                "--num_clients",
                f"{NUM_CLIENTS}",
            ],
            block=False,
            colocate=True,
        )

    # Start FLWR orchestrator
    if (ENV_ID in ["EnergyBalanceModel-v2"]) or (
        ENV_ID in ["EnergyBalanceModel-v3"] and not INFERENCE
    ):
        flwr_model = create_and_start_model(
            exp,
            "FLWR_Orchestrator",
            PYTHON_EXE,
            [f"{BASE_DIR}/{FLWR_EXE}", "--num_clients", f"{NUM_CLIENTS}"],
            block=False,
        )

        # Wait for FLWR process to complete
        wait_for_completion(exp, [flwr_model], label="FLWR")

    # Start the RL algorithms in inference mode
    elif ENV_ID in ["EnergyBalanceModel-v3"] and INFERENCE:
        infx_models = []
        for cid in range(NUM_CLIENTS):
            if GLOBAL:
                infx_model = create_and_start_model(
                    exp,
                    f"infxG_{RL_ALGO}_torch_{SEED}_{cid}",
                    PYTHON_EXE,
                    [
                        f"{BASE_DIR}/fedrl/inference_global.py",
                        f"--env_id {ENV_ID}",
                        f"--algo {RL_ALGO}",
                        f"--optim_group {OPTIM_GROUP}",
                        f"--wandb_group {WANDB_GROUP}",
                        f"--flwr_client {cid}",
                        f"--seed {SEED}",
                        "--capture_video",
                        "--num_steps 200",
                        "--record_step 20000",
                    ],
                    block=False,
                )
                # print(exp.preview(infx_model, verbosity_level="developer"), flush=True)
            else:
                infx_model = create_and_start_model(
                    exp,
                    f"infx_{RL_ALGO}_torch_{SEED}_{cid}",
                    PYTHON_EXE,
                    [
                        f"{BASE_DIR}/rl-algos/inference.py",
                        f"--env_id {ENV_ID}",
                        f"--algo {RL_ALGO}",
                        f"--optim_group {OPTIM_GROUP}",
                        f"--wandb_group {WANDB_GROUP}",
                        f"--flwr_client {cid}",
                        f"--seed {SEED}",
                        "--capture_video",
                        "--num_steps 200",
                        "--record_step 20000",
                    ],
                    block=False,
                )
            # print(exp.preview(infx_model, verbosity_level="developer"), flush=True)
            infx_models.append(infx_model)

        # Wait for RL algorithms to complete
        wait_for_completion(
            exp, infx_models, label=f"{RL_ALGO.upper()}_INFERENCE"
        )

    # Stop all processes after completion
    if ENV_ID in ["EnergyBalanceModel-v3"]:
        exp.stop(ebm_model)

    exp.stop(redis_db)
    print("Experiment completed successfully.", flush=True)


if __name__ == "__main__":
    main()
