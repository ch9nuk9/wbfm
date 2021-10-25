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
source /home/charles/anaconda3/etc/profile.d/conda.sh
#conda activate my_env


CODE_PATH="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts"
SUFFIX="with project_path=$project_path"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path"
else
  conda activate segmentation
  CMD="${CODE_PATH}/2a-pairwise_match_sequential_frames.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/2b-postprocess_matches_to_tracklets.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/2c-reindex_segmentation_training_masks.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/2d-save_training_tracklets_as_dlc.py"
  python $CMD $SUFFIX

  conda activate torch
  CMD="${CODE_PATH}/alternate/3-track_using_fdnc.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/postprocessing/3c+combine_tracklets_and_dlc_tracks.py"
  python $CMD $SUFFIX

  conda activate segmentation
  CMD="${CODE_PATH}/4a-match_tracks_and_segmentation.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/4b-reindex_segmentation_full.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/4c-extract_full_traces.py"
  python $CMD $SUFFIX
fi
