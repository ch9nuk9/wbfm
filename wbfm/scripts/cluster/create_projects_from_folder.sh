#!/bin/bash

# Get all user flags
while getopts t:p:n: flag
do
    case "${flag}" in
        t) DATA_PATH=${OPTARG};;
        p) PROJECT_DIR=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Actually run
COMMAND="/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/0a-create_new_project.py"

# Loop through the parent folder, then try to get the config file within each of these parent folders
for f in "$DATA_PATH"/*; do
    if [[ -d "$f" ]] && [[ "$f" == *"_worm"* ]]; then
        echo "Checking folder: $f"
        EXPERIMENTER=$(cd "$f" && pwd)
        EXPERIMENTER=$(basename "$EXPERIMENTER")
        if [ "$is_dry_run" ]; then
            echo "DRYRUN: Dispatching on folder: $f with EXPERIMENTER: $EXPERIMENTER"
        else
            echo "Dispatching on folder: $f with EXPERIMENTER: $EXPERIMENTER"
            python $COMMAND with project_dir="$PROJECT_DIR" experimenter="$EXPERIMENTER" parent_data_folder="$f" &
        fi
    fi
done

echo "Dispatched all jobs; they will finish in ~30 seconds if successful"
echo "Expected message: INFO - 0a-create_new_project - Completed after 0:00:XX"
