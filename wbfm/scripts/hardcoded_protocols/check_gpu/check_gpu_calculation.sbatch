#!/usr/bin/env bash

#SBATCH --job-name=gpu_check
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=short             # Time limit hrs:min:sec
#SBATCH --output=log_gpu_test-%j.out    # Standard output and error log
#SBATCH --time=0:10:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1  # Does this actually reserve one?
#SBATCH --mem=10G

CMD_DIR="/scratch/neurobiology/zimmer/fieseler/github_repos/dlc_for_wbfm/wbfm/scripts"
CMD="${CMD_DIR}/hardcoded_protocols/check_gpu/check_gpu_calculation.py"

python $CMD
