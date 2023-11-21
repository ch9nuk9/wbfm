#!/bin/bash

# For both gcamp and immobilized projects, first build and save the new visualizations, then copy them to the folder

# Commands
OUTPUT_FOLDER="/scratch/neurobiology/zimmer/wbfm/SummaryPlots/gcamp"
CODE_FOLDER="/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"
BUILD_VIS_COMMAND=$CODE_FOLDER/"scripts/hardcoded_protocols/build_visualizations_for_paper_folders.sh"
COPY_VIS_COMMAND=$CODE_FOLDER/"scripts/visualization/copy_visualizations_for_paper_folders.sh"

# First step: build visualizations (sbatch jobs)
# This script waits for all the jobs to finish
echo "Building visualizations (may take a while)"
bash $BUILD_VIS_COMMAND

# Second step: copy visualizations (need to copy each file type)
# Copy files with multiple extensions
EXTENSIONS=("png" "html")
FILENAMES=("summary_trace_plot" "summary_behavior_plot_kymograph")

for EXT in "${EXTENSIONS[@]}"; do
    for FILENAME in "${FILENAMES[@]}"; do
        bash $COPY_VIS_COMMAND -t 4-traces/"$FILENAME"."$EXT" -d "$OUTPUT_FOLDER"
    done
done
