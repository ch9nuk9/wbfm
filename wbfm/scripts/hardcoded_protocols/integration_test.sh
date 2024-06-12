#!/bin/bash

# Clean and run the integration test, i.e. a shortened dataset
PARENT_PROJECT_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/test_projects"
PARENT_DATA_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/test_data"
CODE_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"

# For each subfolder, remove any project folders
SUBFOLDERS=("freely_moving" "immobilized")
for f in "${SUBFOLDERS[@]}"; do
    PROJECT_PATH=$PARENT_PROJECT_DIR/$f
    if [ -d "$PROJECT_PATH" ]; then
        echo "Removing existing project in subfolder $PROJECT_PATH"
        for SUBFOLDER in "$PROJECT_PATH"/*; do
            if [ -d "$SUBFOLDER" ]; then
                rm -r "$SUBFOLDER"
            fi
        done
    fi
done

# Initialize projects, again for each subfolder
COMMAND=$CODE_DIR/"scripts/cluster/create_multiple_projects_from_data_parent_folder.sh"
for f in "${SUBFOLDERS[@]}"; do
    DATA_DIR=$PARENT_DATA_DIR/$f
    PROJECT_PATH=$PARENT_PROJECT_DIR/$f
    echo "Creating new project in subfolder $PROJECT_PATH"
    # If there are multiple data folders, create a project for each
    bash $COMMAND -t "$DATA_DIR" -p "$PROJECT_PATH" -b False
done

# Sleep, waiting for the projects to be created
echo "Projects should have been created... starting to run the integration test"

# Run, again for each subfolder
# Note that the actual step is different for the immobilized data
COMMAND=$CODE_DIR/"scripts/cluster/run_all_projects_in_parent_folder.sh"
# Freely moving
PROJECT_PATH=$PARENT_PROJECT_DIR/"freely_moving"
bash $COMMAND -t "$PROJECT_PATH" -s traces_and_behavior
# Immobilized
PROJECT_PATH=$PARENT_PROJECT_DIR/"immobilized"
bash $COMMAND -t "$PROJECT_PATH" -s traces
