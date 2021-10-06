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

while getopts t:n: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

CMD="3c-make_full_tracks.sh"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD $project_path"
else
  sbatch $CMD "$project_path"
fi
