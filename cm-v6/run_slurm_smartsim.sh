#!/bin/sh

#SBATCH --job-name=pn341_smartsim_slurm
#SBATCH --output=/gws/nopw/j04/ai4er/users/pn341/climate-rl-f2py/cm-v6/slurm/smartsim_slurm_%j.out
#SBATCH --error=/gws/nopw/j04/ai4er/users/pn341/climate-rl-f2py/cm-v6/slurm/smartsim_slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:10:00
#SBATCH --partition=test

BASE_DIR=/gws/nopw/j04/ai4er/users/pn341/climate-rl-f2py/cm-v6
LOG_DIR="$BASE_DIR/slurm"

set -x

# Checking the conda environment
echo "PYTHON: $(which python)"

# Running the test smartsim script
python -u $BASE_DIR/run_smartsim.py
