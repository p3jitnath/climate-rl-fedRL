<table>
  <tr align="center">
    <!-- UKRI Logo -->
    <td align="center">
      <a href="https://www.ukri.org/">
      <img src="assets/ukri-logo-coloured.png" alt="UKRI Logo" width="400" /></a>
    </td>
    <!-- University of Cambridge Logo -->
    <td align="center">
      <a href="https://www.cam.ac.uk/">
      <img src="assets/cambridge-logo-coloured.png" alt="University of Cambridge" width="400" /> </a>
    </td>
    <!-- Met Office Logo -->
    <td align="center">
      <a href="https://www.metoffice.gov.uk/">
      <img src="assets/met-office-logo-coloured.png" alt="Met Office Logo" width="400" /> </a>
    </td>
  </tr>
</table>


# FedRAIN-Lite: Federated Reinforcement Algorithms for Improving Idealised Numerical Weather and Climate Models
[![arXiv](https://img.shields.io/badge/cs.LG-2508.14315-b31b1b?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2508.14315) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17116349.svg)](https://doi.org/10.5281/zenodo.17116349) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This GitHub repository contains the code, data, and figures for the paper [**FedRAIN-Lite: Federated Reinforcement Algorithms for Improving Idealised Numerical Weather and Climate Models**](https://arxiv.org/abs/2508.14315). Also includes the EBM experiments from the paper [**Learning State-Dependent Weather and Climate Model Parametrisations with Reinforcement Learning**](#).

## Overview

Sub-grid parameterisations in climate models are usually static and offline-tuned, reducing adaptability to changing states. We propose FedRAIN-Lite, a federated RL framework that assigns agents to latitude bands with periodic global aggregation, tested across simplified energy-balance models (ebm-v1 to ebm-v3). Results show Deep Deterministic Policy Gradient (DDPG) consistently outperforms baselines, offering scalable, geographically adaptive parameter learning for future GCMs.

## Project Structure

```
climate-rl-fedRL/
│
├── assets/                 # README.md assets
├── fedrl/                  # Code for fedRL exchange and global inference
├── fedrl-climate-envs/     # Gymansium-based climate environments used in the project
├── flwr/                   # Code for flwr server and clients
├── datasets/               # Dataset files used in simulations
├── misc/                   # Script files for batch-processing runs on JASMIN
├── notebooks/              # Jupyter notebooks for data analysis and results visualization
├── param_tune/             # Code for Ray-powered parameter tuning via multiple parallel batch jobs
├── results/                # Results (imgs and tables) for documentation
├── rl-algos/               # cleanRL-styled source code for RL models
├── smartsim/               # Scripts for running simulations via smartsim
├──.editorconfig            # Config file for code editor specific settings
├──.gitignore               # Config file to skip certain files from version control
├──.pre-commit-config.yaml  # Config file for pre-commit tools for maintaining code quality
├── environment.yml         # Conda environment file
└── pyproject.toml          # Config file for python project
```

## Environment Setup

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/p3jitnath/climate-rl-fedRL.git
   cd climate-rl-fedRL
   ```

2. Install dependencies:
   - Using Conda (recommended):
     ```bash
     conda env create -f environment.yml
     conda activate venv
     ```

3. Replace the `BASE_DIR` location and the conda environment name:
   ```
   find . -type f -exec sed -i "s|/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedRL|$(pwd)|g" {} +
   ```

4. Install the climate RL environments:
    ```
    cd fedrl-climate-envs/ && pip install . && cd ../
    ```

5. [Optional] Download runs:
    ```bash
    wget https://zenodo.org/records/17116349/files/ebm_runs_2025-09-13.zip
    unzip -qq ebm_runs_2025-09-13.zip
    rm -rf ebm_runs_2025-09-13.zip
    ```

## Usage

1. To run an RL algorithm (for eg. DDPG) with an environment (for eg. `EnergyBalanceModel-v1`) over 20000 timesteps with 200 steps in each episode.
```
python ./rl-algos/ddpg/main.py --env_id "EnergyBalanceModel-v1" --total_timesteps 20000 --num-steps 200
```
> [!NOTE]
> Max. value for `num-steps` can be found [here](/fedrl-climate-envs/fedrl_climate_envs/__init__.py).

2. For detailed information regarding each option passed to the algorithm use `-h`.
```
python ./rl-algos/ddpg/main.py -h
```
> [!NOTE]
> Command line examples to run RL algorithms (w/ FedRL) using SLURM can be found in `run` files [here](/misc/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Funding

P. Nath was supported by the [UKRI Centre for Doctoral Training in Application of Artificial Intelligence to the study of Environmental Risks](https://ai4er-cdt.esc.cam.ac.uk/) [EP/S022961/1].

## Contact

For any queries or further information, please contact [Pritthijit Nath](mailto:pn341@cam.ac.uk).
