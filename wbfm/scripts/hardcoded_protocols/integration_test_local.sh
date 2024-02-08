#!/bin/bash

# Clean and run the integration test, i.e. a shortened dataset
PROJECT_DIR="/scratch/neurobiology/zimmer/fieseler/wbfm_projects/project_pytest"
CODE_DIR="/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"

# Clean
COMMAND=$CODE_DIR/"scripts/postprocessing/delete_analysis_files.py"
PROJECT_PATH=$PROJECT_DIR/"project_config.yaml"
python $COMMAND with project_path=$PROJECT_PATH dryrun=False

# Run using the snakemake pipeline from a bash (would be sbatch on the cluster) controller job
cd $PROJECT_DIR/snakemake || exit
COMMAND="RUNME.sh -c"

bash $COMMAND
