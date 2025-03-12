# 0. Perform cleanup
rm -rf SM-FLWR_Orchestrator

# 1. Run smartsim
cd f2py-climate-envs && pip install . && cd ..
sbatch slurm_smartsim.sh
