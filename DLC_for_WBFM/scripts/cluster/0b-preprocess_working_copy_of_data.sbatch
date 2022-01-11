#!/usr/bin/env bash

#SBATCH --job-name=preprocessing_and_zarr_conversion
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

pwd; hostname; date

while getopts t:n: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done


CMD="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts/0a-create_new_project.py"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path"
else
  python $CMD with project_path="$project_path"
fi