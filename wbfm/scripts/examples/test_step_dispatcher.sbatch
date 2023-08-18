#!/usr/bin/env bash

#SBATCH --job-name=test_step
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=short             # Time limit hrs:min:sec
#SBATCH --output=log_test_step-%j.out    # Standard output and error log
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

pwd; hostname; date
#module load cudnn/7.6.5

# Necessary because the default home directory is not writable from the cluster
export HOME="/scratch/neurobiology/zimmer/Charles"

# Get all user flags
while getopts t:n:s: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        s) step_reference=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Unpack the flag into an actual command
CMD_DIR="/scratch/neurobiology/zimmer/fieseler/github_repos/dlc_for_wbfm/wbfm/scripts"

declare -A CMD_DICT
declare -A CMD_DICT
CMD_DICT["0b"]="0b-preprocess_working_copy_of_data.py"
CMD_DICT["1"]="1-segment_video.py"
CMD_DICT["2a"]="2a-pairwise_match_sequential_frames.py"
CMD_DICT["2a-alt"]="alternate/2+only_build_frame_objects.py"
CMD_DICT["2b"]="2b-postprocess_matches_to_tracklets.py"
CMD_DICT["2c"]="2c-reindex_segmentation_training_masks.py"
CMD_DICT["3a"]="3a-track_using_fdnc.py"
CMD_DICT["3a-alt"]="3a-track_using_embedding.py"
CMD_DICT["3a-superglue"]="3a-track_using_superglue.py"
CMD_DICT["3b"]="3b-match_tracklets_and_tracks_using_neuron_initialization.py"
CMD_DICT["4"]="4-make_final_traces.py"

CMD="${CMD_DIR}/${CMD_DICT[$step_reference]}"
FULL_CMD="python $CMD with project_path=$project_path"

echo "Running step $step_reference using full command:"
echo $FULL_CMD

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $FULL_CMD"
else
  python $CMD with project_path="$project_path"
fi

date
echo "Finished; check logs to determine success"
