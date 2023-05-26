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

# The actual folder will be zipped, with the above path as the base name but with an additional number suffix
# First, find the exact folder and unzip it. Then continue with moving individual files.
zipped_excel_folder=$(find "$excel_folder" -type f -name "*.zip")
if [ -f "$zipped_excel_folder" ]; then
    echo "Found zipped excel folder: $zipped_excel_folder"

    # Unzip the folder
    unzip "$zipped_excel_folder" -d "$excel_folder"

    # Remove the zipped folder
    rm "$zipped_excel_folder"
else
    echo "Could not find zipped excel folder: $zipped_excel_folder"
    echo "You must manually download from google drive!"
    # Exit with error
    exit 1
fi

# The unzipped folder may have a subfolder called "wbfm_neuron_ids"; if so we need to update excel_folder
unzipped_excel_folder=$(find "$excel_folder" -type d -name "wbfm_neuron_ids")
if [ -d "$unzipped_excel_folder" ]; then
    echo "Found nested unzipped excel folder: $unzipped_excel_folder"
    # Update the excel folder
    excel_folder="$unzipped_excel_folder"
else
    echo "Excel folder was not nested, and remains: $excel_folder"
fi

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
