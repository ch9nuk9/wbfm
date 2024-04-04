#!/bin/bash

# Create new projects from a folder of data subfolders
# This script is meant to be run on the cluster, or on a local machine with the data mounted
# Note that if you initialize projects on a non-linux machine, you may need to change the paths to the data and
# project directories (thus cluster is suggested)
#
# Usage:
#   bash create_multiple_projects_from_data_parent_folder.sh -t <DATA_PARENT_FOLDER> -p <PROJECT_PARENT_FOLDER> -n <is_dry_run>
#
# Example:
#   bash create_multiple_projects_from_data_parent_folder.sh -t /scratch/neurobiology/zimmer/wbfm/data -p /scratch/neurobiology/zimmer/wbfm/projects

# Get all user flags
while getopts t:p:n: flag
do
    case "${flag}" in
        t) DATA_PARENT_FOLDER=${OPTARG};;
        p) PROJECT_PARENT_FOLDER=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Actually run
COMMAND="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/0a-create_new_project.py"

# Loop through the parent folder, then try to get the config file within each of these parent folders
# Counter for number of jobs actually submitted
num_jobs=0
for f in "$DATA_PARENT_FOLDER"/*; do
    if [[ -d "$f" ]] && [[ "$f" == *"worm"* ]]; then
        echo "Checking folder: $f"
        num_jobs=$((num_jobs+1))
        EXPERIMENTER=$(cd "$f" && pwd)
        EXPERIMENTER=$(basename "$EXPERIMENTER")
        if [ "$is_dry_run" ]; then
            echo "DRYRUN: Dispatching on folder: $f with EXPERIMENTER: $EXPERIMENTER"
        else
            echo "Dispatching on folder: $f with EXPERIMENTER: $EXPERIMENTER"
            python $COMMAND with project_dir="$PROJECT_PARENT_FOLDER" experimenter="$EXPERIMENTER" parent_data_folder="$f" &
        fi
    fi
done

echo "===================================================================================="
echo "Dispatched $num_jobs jobs in the background; they will finish in ~30 seconds if successful"
echo "Note that the jobs will print out their progress as they complete, and will mix messages"
echo "Expected message if successful:"
echo "INFO - 0a-create_new_project - Completed after 0:00:XX"
echo "===================================================================================="
