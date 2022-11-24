#!/bin/bash

# Get all user flags
while getopts t:p: flag
do
    case "${flag}" in
        t) DATA_PATH=${OPTARG};;
        p) PROJECT_DIR=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

PARENT_DATA_FOLDER=$(find "$DATA_PATH" -maxdepth 1 -mindepth 1 -type d -wholename "*worm*" | head -n "$SLURM_ARRAY_TASK_ID" | tail -n 1)

# Actually run
COMMAND="/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/0a-create_new_project.py"

#
EXPERIMENTER=$(basename "$PARENT_DATA_FOLDER")
python $COMMAND with project_dir="$PROJECT_DIR" parent_data_folder="$PARENT_DATA_FOLDER" experimenter="$EXPERIMENTER"

echo "Finished; check logs to determine success"
