#!/usr/bin/env bash

#SBATCH --job-name=fdnc_tracking
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --partition=gpu
# #SBATCH --cpus-per-task=8

pwd; hostname; date

while getopts t:n: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

CMD="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts/alternate/3a-track_using_random_fdnc.py"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path"
else
  python $CMD with project_path="$project_path"
fi
