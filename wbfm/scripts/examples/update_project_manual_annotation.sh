#!/usr/bin/env bash

# This script moves an excel file with manual annotations to the project folder
# Start point:
# - Folder of excel files with manual annotations
# - Folder of folder of projects
# - The excel file has the same name as the project folder
#
# End point:
# - The excel file is moved to the project folder, to the 3-tracking/manual_annotation folder
#

# Set the path to the folder with the excel files
excel_folder="/home/charles/Downloads/wbfm_neuron_ids"

# Set the path to the folder with the projects
project_parent_folder="/scratch/neurobiology/zimmer/Charles/dlc_stacks"

# Loop over all project parent folders (actually projects are subfolders within the parent folder)
for project_parent in "$project_parent_folder"/*; do
    if [ -d "$project_parent" ] && [ ! -L "$project_parent" ]; then
        echo "Checking folder: $project_parent"

        # Loop over all projects within the parent folder
        for project in "$project_parent"/*; do
            if [ -d "$project" ] && [ ! -L "$project" ]; then
                echo "Checking folder: $project"

                # Get the name of the project
                project_name=$(basename "$project")

                # Get the path to the excel file
                excel_file="$excel_folder/$project_name.xlsx"

                # Check if the excel file exists
                if [ -f "$excel_file" ]; then
                    echo "Found excel file: $excel_file"

                    # Get the path to the manual annotation folder
                    manual_annotation_folder="$project/3-tracking/manual_annotation"

                    # Check if the manual annotation folder exists
                    if [ -d "$manual_annotation_folder" ]; then
                        echo "Found manual annotation folder: $manual_annotation_folder"

                        # Move the excel file to the manual annotation folder
                        mv "$excel_file" "$manual_annotation_folder"
                    else
                        echo "Could not find manual annotation folder: $manual_annotation_folder"
                    fi
                else
                    echo "Could not find excel file: $excel_file"
                fi
            fi
        done
    fi
done
