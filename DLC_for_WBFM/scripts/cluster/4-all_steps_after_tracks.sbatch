#!/usr/bin/env bash

#SBATCH --job-name=make_traces   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=make_traces.out    # Standard output and error log
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --mem=16G

pwd; hostname; date

while getopts t:n: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

CODE_PATH="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts"
SUFFIX="with project_path=$project_path"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path"
else
  CMD="${CODE_PATH}/4a-match_tracks_and_segmentation.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/4b-reindex_segmentation_full.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/4c-extract_full_traces.py"
  python $CMD $SUFFIX
fi
