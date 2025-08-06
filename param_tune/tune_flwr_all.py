#
# RUN: nohup python $PWD/param_tune/tune_flwr_all.py > out.log 2>&1 &
#

import re
import subprocess
import time

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"

sbatch_cmds = [
    # f'sbatch {BASE_DIR}/param_tune/tune_flwr_slurm.sh --algo td3 --exp_id "ebm-v3-fedRL-L-20k-a6-fed05" --env_id "EnergyBalanceModel-v3" --opt_timesteps 20000',
    # f'sbatch {BASE_DIR}/param_tune/tune_flwr_slurm.sh --algo ddpg --exp_id "ebm-v3-fedRL-L-20k-a6-fed05" --env_id "EnergyBalanceModel-v3" --opt_timesteps 20000',
]


def submit_and_wait(cmd):
    print(f"Submitting: {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error submitting job: {result.stderr}", flush=True)
        return

    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        print("Failed to extract job ID.")
        return

    job_id = match.group(1)
    print(f"Submitted job {job_id}. Waiting for completion...", flush=True)

    while True:
        check = subprocess.run(
            ["squeue", "-j", job_id], capture_output=True, text=True
        )
        if job_id not in check.stdout:
            print(f"Job {job_id} finished.", flush=True)
            break
        time.sleep(30)


for cmd in sbatch_cmds:
    submit_and_wait(cmd)
