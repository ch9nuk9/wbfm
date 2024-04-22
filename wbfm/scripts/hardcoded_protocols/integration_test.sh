#!/bin/bash

# Clean and run the integration test, i.e. a shortened dataset
PARENT_PROJECT_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/test_projects"
CODE_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"

# Clean
COMMAND=$CODE_DIR/"scripts/postprocessing/delete_analysis_files.py"
# Loop through each subfolder in the parent project directory
for f in "$PARENT_PROJECT_DIR"/*; do
    if [ -d "$f" ] && [ ! -L "$f" ]; then
        echo "Checking folder: $f"
        PROJECT_PATH=$f/"project_config.yaml"
        python $COMMAND with project_path=$PROJECT_PATH dryrun=False
    fi
done

# Run using my helper script for all in folder
COMMAND=$CODE_DIR/"scripts/cluster/run_all_projects_in_parent_folder.sh"
bash $COMMAND -t $PARENT_PROJECT_DIR -s traces_and_behavior

