#!/bin/bash
# Opens tmux session and runs snakemake for all projects in a folder. Example dry run usage:
# bash run_all_projects_in_parent_folder.sh -t '/path/to/parent/folder' -n True
#
# For real usage, remove '-n True' and update the path after -t
#

# Add help function
function usage {
  echo "Usage: $0 [-t folder_of_projects] [-n] [-d] [-s rule] [-h]"
  echo "  -t: folder of projects (required)"
  echo "  -n: dry run of this script (default: false)"
  echo "  -d: dry run of snakemake (default: false)"
  echo "  -s: snakemake rule to run (default: traces_and_behavior; other options: traces, behavior)"
  echo "  -h: display help (this message)"
  exit 1
}

RULE="traces_and_behavior"
is_dry_run=""
RUNME_ARGS=""

# Get all user flags
while getopts t:n:s:d:ch flag
do
    case "${flag}" in
        t) folder_of_projects=${OPTARG};;
        n) is_dry_run="True";;
        d) is_snakemake_dry_run=${OPTARG};;
        c) RUNME_ARGS="-c";;
        s) RULE=${OPTARG};;
        h) usage;;
        *) raise error "Unknown flag"
    esac
done

# Shared setup for each command
conda_setup_cmd="conda activate /lisc/scratch/neurobiology/zimmer/.conda/envs/wbfm/"

# Loop through the parent folder, then try to get the config file within each of these parent folders
for f in "$folder_of_projects"/*; do
    if [ -d "$f" ] && [ ! -L "$f" ]; then
        echo "Checking folder: $f"

        for f_config in "$f"/*; do
            if [ -f "$f_config" ] && [ "${f_config##*/}" = "project_config.yaml" ]; then
                if [ "$is_dry_run" ]; then
                    # Run the snakemake dryrun
                    echo "DRYRUN: Dispatching on config file: $f_config"
                else
                    # Get the snakemake command and run it
                    snakemake_folder="$f/snakemake"
                    snakemake_script_path="$snakemake_folder/RUNME.sh"

                    snakemake_cmd="$snakemake_script_path -s $RULE $RUNME_ARGS"
                    if [ "$is_snakemake_dry_run" ]; then
                       snakemake_cmd="$snakemake_cmd -n"
                       echo "Running snakemake dry run"
                    fi
                    # Instead of tmux, use a controller sbatch job
                    cd "$snakemake_folder" || exit  # Move in order to create the snakemake log all together

                    # Build the job name using the folder name and the target rule
                    JOB_NAME="${f}_${RULE}"
                    echo "Running job with name: $JOB_NAME"

                    # If the RUNME_ARGS contains -c, then run the command directly without sbatch
                    if [ "$RUNME_ARGS" = "-c" ]; then
                        # Do not run the conda setup command, which is not needed for local runs
                        echo "Running: $snakemake_cmd"
                        bash $snakemake_cmd &
                    else
                        full_cmd="$conda_setup_cmd; bash $snakemake_cmd"
                        sbatch --time 5-00:00:00 \
                            --cpus-per-task 1 \
                            --mem 1G \
                            --mail-type=FAIL,TIME_LIMIT,END \
                            --wrap="$full_cmd" \
                            --job-name="$JOB_NAME"
                    fi
                fi
            fi
        done
    fi
done
