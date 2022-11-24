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

PARENT_DATA_FOLDER=$(find "$DATA_PATH" -maxdepth 1 -mindepth 1 -type d -wholename "*worm*" | head -n "$SLURM_ARRAY_TASK_ID" | tail -n 1)

# Actually run
COMMAND="/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/0a-create_new_project.py"
EXPERIMENTER=$(basename "$PARENT_DATA_FOLDER")
FULL_COMMAND=$COMMAND with project_dir="$PROJECT_DIR" experimenter="$EXPERIMENTER"

# Loop through the parent folder, then try to get the config file within each of these parent folders
for f in "$DATA_PATH"/*; do
    if [ -d "$f" ] && [ ! -L "$f" ]; then
        echo "Checking folder: $f"

        for subfolder in "$f"/*; do
            if [[ -d "$subfolder" ]] && [[ "$subfolder" == *"_worm"* ]]; then
                if [ "$is_dry_run" ]; then
                    echo "DRYRUN: Dispatching on folder: $subfolder"
                else
                    python $COMMAND
                fi
            fi
        done
    fi
done

python $COMMAND with project_dir="$PROJECT_DIR" parent_data_folder="$PARENT_DATA_FOLDER" experimenter="$EXPERIMENTER"

echo "Finished; check logs to determine success"
