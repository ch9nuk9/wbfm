#!/bin/bash

# Loop through folders, and copy a target png to a destination folder
# Assumed structure:
#   - parent_folder
#     - project_folder_1
#       - visualization
#         - target.png
#     - project_folder_2 (new name)
#       - visualization (same name)
#         - target.png (same name)
#
# Usage:
#   ./copy_visualizations_from_many_projects.sh -s source_folder -t visualization/target.png -d destination_folder
#
# Example:
#   bash copy_visualizations_from_many_projects.sh -s/lisc/scratch/neurobiology/zimmer/Charles/dlc_stacks/2022-11-27_spacer_7b_2per_agar -t 4-traces/summary_trace_plot_kymograph.png -d/lisc/scratch/neurobiology/zimmer/Charles/multiproject_visualizations/gcamp_kymograph
#
# Note that the target does not need to be in a subfolder called "visualization"

# Parse arguments
while getopts s:t:d: flag
do
    case "${flag}" in
        s) source_folder=${OPTARG};;
        t) target=${OPTARG};;
        d) destination_folder=${OPTARG};;
        *) echo "Invalid option";;
    esac
done

# Make the destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Loop through folders
for folder in "$source_folder"/*; do
    # Get project name
    project_name=$(basename "$folder")
    # Get target png
    target_png=$folder/$target
    # Get the extension of the target, which may be .png, .html, etc.
    target_extension="${target##*.}"
    # Copy target png to destination folder, renaming it to the project name
    cp "$target_png" "$destination_folder"/"$project_name"."$target_extension"
done
