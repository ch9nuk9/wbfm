#!/usr/bin/env bash

#SBATCH --job-name=superglue_training
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=long             # Time limit hrs:min:sec
#SBATCH --output=log_superglue_training-%j.out    # Standard output and error log
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --mail-user=charles.fieseler@univie.ac.at
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Necessary because the default home directory is not writable from the cluster
export HOME="/scratch/neurobiology/zimmer/Charles"

CMD_DIR="/scratch/neurobiology/zimmer/fieseler/github_repos/dlc_for_wbfm/wbfm/scripts/nn_training"
CMD="${CMD_DIR}/train_superglue.py"

python $CMD
