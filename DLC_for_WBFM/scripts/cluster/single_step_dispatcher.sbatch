#!/usr/bin/env bash

#SBATCH --job-name=single_step
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=log_single_step.out    # Standard output and error log
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --partition=gpu

pwd; hostname; date
module load cudnn/7.6.5

while getopts t:n:s: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        s) step_reference=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

CMD_DIR="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts"
CMD_DIR="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts"

declare -A CMD_DICT
CMD_DICT["2a"]="2a-pairwise_match_sequential_frames.py"
CMD_DICT["2b"]="2b-postprocess_matches_to_tracklets.py"
CMD_DICT["2c"]="2c-reindex_segmentation_training_masks.py"
CMD_DICT["3a"]="3a-track_using_fdnc.py"
CMD_DICT["3b"]="3b-match_tracklets_and_dlc_tracks.py"
CMD_DICT["3c"]="3c-make_final_tracks_using_tracklet_matches.py"

CMD="${CMD_DIR}/${CMD_DICT[$step_reference]}"

echo "Running step $step_reference using full command:"
echo $CMD

LOG="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts/cluster/fdnc_tracking.log"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD with project_path=$project_path" > $LOG
else
  python $CMD with project_path="$project_path"
fi

date
echo "Finished"
