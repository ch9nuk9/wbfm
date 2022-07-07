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

CODE_PATH="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/wbfm/scripts"
SUFFIX="with project_path=$project_path"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path"
else


  # Initially start from manual matches
  CMD="${CODE_PATH}/3b-match_tracklets_and_dlc_tracks.py"
  python $CMD $SUFFIX start_from_manual_matches=True use_imputed_df=True
  # Now there are new matches
  # TODO: need to properly use wiggle-modified tracklets
  CMD="${CODE_PATH}/3c-make_final_tracks_using_tracklet_matches.py"
  python $CMD $SUFFIX start_from_manual_matches=False use_imputed_df=True

  CMD="${CODE_PATH}/4a-match_tracks_and_segmentation.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/4b-reindex_segmentation_full.py"
  python $CMD $SUFFIX
  CMD="${CODE_PATH}/4c-extract_full_traces.py"
  python $CMD $SUFFIX
fi
