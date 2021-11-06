#!/usr/bin/env bash

#SBATCH --job-name=combine_tracks
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

pwd; hostname; date

while getopts t:n: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

CMD="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts/3b-combine_tracklets_and_dlc_tracks.py"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path"
else
  python $CMD with project_path="$project_path"
fi
