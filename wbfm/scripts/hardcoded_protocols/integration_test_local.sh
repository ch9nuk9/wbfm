#!/bin/bash

# Clean and run the integration test, i.e. a shortened dataset
DATA_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/example_data/freely_moving/ZIM2165_Gcamp7b_worm1"
PROJECT_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/test_projects"
CODE_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"

# Initialize project
COMMAND=$CODE_DIR/"scripts/0a-create_new_project.py"
COMMAND="$COMMAND with parent_data_folder=$DATA_DIR project_dir=$PROJECT_DIR experimenter=pytest"
python "$COMMAND"

# Get the newest folder using find and assume that is the one that was just created
NEWEST_PROJECT_DIR=$(find $PROJECT_DIR -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

# Clean
#COMMAND=$CODE_DIR/"scripts/postprocessing/delete_analysis_files.py"
#PROJECT_PATH=$PROJECT_DIR/"project_config.yaml"
#python $COMMAND with project_path=$PROJECT_PATH dryrun=False

# Run using the snakemake pipeline from a bash (would be sbatch on the cluster) controller job
cd "$NEWEST_PROJECT_DIR"/snakemake || exit
COMMAND="RUNME.sh -c"

bash "$COMMAND"
