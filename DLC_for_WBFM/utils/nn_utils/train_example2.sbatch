#!/usr/bin/env bash

#SBATCH --job-name=leifer_pre_training2   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8
#SBATCH --qos=short             # Time limit hrs:min:sec
#SBATCH --output=leifer_pre_training2.out    # Standard output and error log
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
# # SBATCH --gpus-per-task=1

pwd; hostname; date

module load cudnn/7.6.5

while getopts t:n: flag
do
    case "${flag}" in
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

CMD="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/utils/nn_utils/train.py"
DATA_PATH="/scratch/zimmer/Charles/github_repos/fDNC_Neuron_ID/Data"
OUTPUT_PATH="/scratch/zimmer/Charles/github_repos/fDNC_Neuron_ID/fDNC/model/cluster"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD"
else
  python $CMD --train_path "${DATA_PATH}/train_synthetic" --eval_path "${DATA_PATH}/validation_synthetic" --model_path "${OUTPUT_PATH}" --cuda_device_index 1
fi
