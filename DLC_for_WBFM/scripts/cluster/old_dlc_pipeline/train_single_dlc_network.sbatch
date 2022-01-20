#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=16G

module load cudnn/7.6.5

echo "Output of nvidia-smi:"
eval "nvidia-smi"

WBFM_DIR="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm"
CMD="${WBFM_DIR}/DLC_for_WBFM/scripts/cluster/train_single_dlc_network.py"
dlc_config=$1
echo "Running on config file: $dlc_config"

log_folder="$(dirname "$dlc_config")"
log_fname="${log_folder}/dlc_training.log"

python $CMD --dlc_config "$dlc_config" > "$log_fname"
