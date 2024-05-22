#!/usr/bin/env bash

#SBATCH --job-name=gpu_step
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=log_gpu_step-%j.out    # Standard output and error log
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=basic  # The basic partition has computers with single GPUs, instead of multiple

pwd; hostname; date

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
CMD_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts"

declare -A CMD_DICT
declare -A CMD_DICT
CMD_DICT["0b"]="0b-preprocess_working_copy_of_data.py"
CMD_DICT["1"]="1-segment_video.py"
CMD_DICT["2a"]="2a-build_frame_objects.py"
CMD_DICT["2b"]="2b-match_adjacent_volumes.py"
CMD_DICT["2c"]="2c-postprocess_matches_to_tracklets.py"
CMD_DICT["3a"]="3a-track_using_superglue.py"
CMD_DICT["3b"]="3b-match_tracklets_and_tracks_using_neuron_initialization.py"
CMD_DICT["3-alt"]="alternate/3-track_using_clustering.py"
CMD_DICT["4"]="4-make_final_traces.py"
CMD_DICT["4-alt"]="alternate/4c-extract_full_traces.py"

# Requires additional path to model, currently hardcoded
CMD_DICT["4+barlow_track"]="postprocessing/4+track_using_barlow.py"

CMD="${CMD_DIR}/${CMD_DICT[$step_reference]}"
FULL_CMD="$CMD with project_path=$project_path"

if [ "$step_reference" = "4+barlow_track" ]; then
  FULL_CMD="$FULL_CMD model_fname=/lisc/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/barlow_ZIM2165_Gcamp7b_worm1-2022_11_28_from_search/trial_0/resnet50.pth"
fi

echo "Running step $step_reference using full command:"
echo "$FULL_CMD"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $FULL_CMD"
else
  python "$FULL_CMD"
fi

date
echo "Finished; check logs to determine success"
