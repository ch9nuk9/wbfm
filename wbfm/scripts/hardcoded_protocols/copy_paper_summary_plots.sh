#!/bin/bash

# Commands
COPY_VIS_COMMAND=$CODE_FOLDER/"scripts/visualization/copy_visualizations_for_paper_folders.sh"
OUTPUT_FOLDER="/scratch/neurobiology/zimmer/wbfm/SummaryPlots/gcamp"

# Copy files with multiple extensions
EXTENSIONS=("png" "html")
FILENAMES=("summary_trace_plot" "summary_behavior_plot_kymograph")

for EXT in "${EXTENSIONS[@]}"; do
    for FILENAME in "${FILENAMES[@]}"; do
        bash "$COPY_VIS_COMMAND" -t 4-traces/"$FILENAME"."$EXT" -d "$OUTPUT_FOLDER"
    done
done