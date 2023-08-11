#!/usr/bin/env bash

#SBATCH --job-name=single_step
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=log_single_step-%j.out    # Standard output and error log
#SBATCH --time=0-12:00:00  # A normal video should only take <3 hours, but some could be longer
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=basic,gpu

pwd; hostname; date

# Necessary because the default home directory is not writable from the cluster
# UPDATE: as of August 2022 this is not necessary
#export HOME="/scratch/neurobiology/zimmer/Charles"

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
CMD_DICT["0b"]="0b-preprocess_working_copy_of_data.py"
CMD_DICT["0-alt"]="alternate/0+zip_raw_zarr.py"
CMD_DICT["1"]="1-segment_video.py"
CMD_DICT["1-alt"]="alternate/1+rebuild_segmentation_metadata.py"
CMD_DICT["2a"]="2a-build_frame_objects.py"
CMD_DICT["2b"]="2b-match_adjacent_volumes.py"
CMD_DICT["2c"]="2c-postprocess_matches_to_tracklets.py"
CMD_DICT["3a"]="3a-track_using_superglue.py"
CMD_DICT["3b"]="3b-match_tracklets_and_tracks_using_neuron_initialization.py"
CMD_DICT["4"]="4-make_final_traces.py"
CMD_DICT["4-alt"]="alternate/4c-extract_full_traces.py"
CMD_DICT["4-pixel"]="visualization/4+extract_pixel_values_for_tracked_neurons.py"
CMD_DICT["4-grid"]="visualization/4+make_summary_grid_plots.py"
CMD_DICT["4-export"]="visualization/4+save_final_dataframe.py"
CMD_DICT["4-trigger"]="visualization/4+make_triggered_average_grid_plot.py"
CMD_DICT["4-summary"]="visualization/4+make_summary_interactive_plot.py"
CMD_DICT["4-encoding"]="folder_analysis/4+make_multiproject_behavioral_encodings.py"
CMD_DICT["check_gpu"]="examples/check_gpu.py"
CMD_DICT["manual"]="postprocessing/3b+update_after_manual_annotation.py"

CMD="${CMD_DIR}/${CMD_DICT[$step_reference]}"
FULL_CMD="python $CMD with project_path=$project_path"

echo "Running step $step_reference using full command:"
echo "$FULL_CMD"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $FULL_CMD"
else
  python "$CMD" with project_path="$project_path"
fi

date
echo "Finished; check logs to determine success"
