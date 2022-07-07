#!/usr/bin/env bash

pwd; hostname; date

while getopts t:n: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Make conda functions available
# https://github.com/conda/conda/issues/7980
#source /home/charles/anaconda3/etc/profile.d/conda.sh
#conda init bash

CODE_PATH="/scratch/neurobiology/zimmer/Charles/github_repos/dlc_for_wbfm/wbfm/scripts"
SUFFIX="with project_path=$project_path"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path"
else

#  conda activate torch
#  source activate torch

  CMD="${CODE_PATH}/3b-match_tracklets_and_tracks_using_neuron_initialization.py"
  python $CMD $SUFFIX

  CMD="${CODE_PATH}/4-make_final_traces.py"
  python $CMD $SUFFIX
fi
