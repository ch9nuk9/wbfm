#!/usr/bin/env bash

#SBATCH --job-name=dlc_tracking   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=dlc_tracking.out    # Standard output and error log
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
# # SBATCH --gpus-per-task=1

pwd; hostname; date

module load cudnn/7.6.5

while getopts t:n: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

CMD="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts/3c-make_full_tracks.py"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path"
else
  python $CMD with project_path="$project_path"
fi
