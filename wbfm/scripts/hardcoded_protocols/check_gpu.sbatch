#!/usr/bin/env bash

#SBATCH --job-name=gpu_check
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=short             # Time limit hrs:min:sec
#SBATCH --output=log_gpu_test-%j.out    # Standard output and error log
#SBATCH --time=0:10:00
#SBATCH --partition=basic
#SBATCH --gres=gpu:1

CMD_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts"
CMD="${CMD_DIR}/hardcoded_protocols/check_gpu.py"

python $CMD
