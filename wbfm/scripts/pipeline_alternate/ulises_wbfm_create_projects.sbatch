#!/bin/bash

#This is the way to run an array job without running two bash scripts

#This file has a copy in /cluster_jobs/

#SBATCH --job-name=wbfm_create_projects
# This job requests from SLURM to allocate 1 node.
#SBATCH --nodes=1
# On that node, it will run 4 tasks, each with 1 core and 1 GB of memory.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# After Thomas Rattei suggestion I specify mem instead of mem-per-cpu
#SBATCH --mem=1G
# Running time will be 3hours
# And it will place the output of the commands into following file
#SBATCH --output=/scratch/neurobiology/zimmer/ulises/code/cluster_outputs/%x_output_%A_%a_.txt.txt
#SBATCH --array=1-15

echo "All jobs in this array have:"
echo "- SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "- SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}"
echo "- SLURM_ARRAY_TASK_MIN=${SLURM_ARRAY_TASK_MIN}"
echo "- SLURM_ARRAY_TASK_MAX=${SLURM_ARRAY_TASK_MAX}"

echo "This job in the array has:"
echo "- SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "- SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

cd/lisc/scratch/neurobiology/zimmer/ulises/wbfm/20221123/data/

PARENT_DATA_FOLDER=$(find $PWD -maxdepth 1 -mindepth 1 -type d -wholename "*worm*" | head -n $SLURM_ARRAY_TASK_ID | tail -n 1)


source /apps/conda/miniconda3/bin/activate/lisc/scratch/neurobiology/zimmer/.conda/envs/wbfm/
echo "Environment loaded?"
# you can run it with dryrun=false first if you want to check what will be deleted

PROJECT_DIR="/scratch/neurobiology/zimmer/ulises/wbfm_projects/"

# Actually run
COMMAND="/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/0a-create_new_project.py"
# Alternative: can just give the overall data path

#
EXPERIMENTER=$(basename $PARENT_DATA_FOLDER)

echo $EXPERIMENTER

python $COMMAND with project_dir=$PROJECT_DIR parent_data_folder=$PARENT_DATA_FOLDER experimenter=$EXPERIMENTER



echo "ended sbatch script"