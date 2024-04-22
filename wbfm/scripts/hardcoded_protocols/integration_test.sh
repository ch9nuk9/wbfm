#!/bin/bash

# Clean and run the integration test, i.e. a shortened dataset
PROJECT_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/test_projects/pytest-raw"
CODE_DIR="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm"

# Clean
COMMAND=$CODE_DIR/"scripts/postprocessing/delete_analysis_files.py"
PROJECT_PATH=$PROJECT_DIR/"project_config.yaml"
python $COMMAND with project_path=$PROJECT_PATH dryrun=False

# Run using the snakemake pipeline from an sbatch controller job
cd $PROJECT_DIR/snakemake || exit
COMMAND="RUNME.sh"

# The bash script also accepts command line arguments, so we want to avoid them being passed to sbatch
sbatch --time 1-00:00:00 \
        --cpus-per-task 1 \
        --mem 1G \
        $COMMAND
