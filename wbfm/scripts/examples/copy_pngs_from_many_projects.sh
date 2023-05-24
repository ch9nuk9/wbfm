#!/bin/bash

# Loop through folders, and copy a target png to a destination folder
# Assumed structure:
#   - parent_folder
#     - project_folder_1
#       - visualization
#         - target.png
#     - project_folder_2 (new name)
#       - visualization (same name; hardcoded)
#         - target.png (same name)
#
# Usage:
#   ./copy_pngs_from_many_projects.sh -s source_folder -t target.png -d destination_folder
#
# Example:
#   ./copy_pngs_from_many_projects.sh -s ~/Desktop/parent_folder -t target.png -d ~/Desktop/destination_folder

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
    # Get visualization folder
    visualization_folder=$folder/visualization
    # Get target png
    target_png=$visualization_folder/$target
    # Copy target png to destination folder, renaming it to the project name
    cp "$target_png" "$destination_folder"/"$project_name".png
done
