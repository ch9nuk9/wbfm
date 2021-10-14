#!/usr/bin/env bash

#SBATCH --job-name=fdnc_tracking   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=fdnc_tracking.out    # Standard output and error log
#SBATCH --time=4:00:00
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

CMD="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts/alternate/3-track_using_fdnc.py"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path"
else
  python $CMD with project_path="$project_path"
fi
