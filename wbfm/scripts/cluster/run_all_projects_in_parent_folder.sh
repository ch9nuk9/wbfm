#!/bin/bash
# Opens tmux session and runs snakemake for all projects in a folder. Example dry run usage:
# bash run_all_projects_in_parent_folder.sh -t '/path/to/parent/folder' -n True
#
# For real usage, remove '-n True' and update the path after -t
#

RULE="traces_and_behavior"
# Get all user flags
while getopts t:n:s:d: flag
do
    case "${flag}" in
        t) folder_of_projects=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        d) is_snakemake_dry_run=${OPTARG};;
        s) RULE=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Loop through the parent folder, then try to get the config file within each of these parent folders
i_tmux=0
for f in "$folder_of_projects"/*; do
    if [ -d "$f" ] && [ ! -L "$f" ]; then
        echo "Checking folder: $f"

        for f_config in "$f"/*; do
            if [ -f "$f_config" ] && [ "${f_config##*/}" = "project_config.yaml" ]; then
                tmux_name="worm$i_tmux"
                if [ "$is_dry_run" ]; then
                    # Run the snakemake dryrun
                    echo "DRYRUN: Dispatching on config file: $f_config with tmux session: $tmux_name"
                else
                    echo "Opening tmux session: $tmux_name"
                    # Get the snakemake command and run it
                    setup_cmd="conda activate /scratch/neurobiology/zimmer/.conda/envs/wbfm/"
                    snakemake_folder="$f/snakemake"
                    if [ "$is_snakemake_dry_run" ]; then
                       snakemake_cmd="$snakemake_folder/RUNME.sh -n -s $RULE"
                       echo "Running snakemake dry run"
                    else
                       snakemake_cmd="$snakemake_folder/RUNME.sh -s $RULE"
                    fi
                    # Check if the session exists... don't see how to do a while loop here, because I'm using $?
                    tmux has-session -t=$tmux_name 2>/dev/null
                    if [ $? = 0 ]; then
                      tmux_name="$tmux_name-1"
                    fi
                    tmux has-session -t=$tmux_name 2>/dev/null
                    if [ $? = 0 ]; then
                      tmux_name="$tmux_name-1"
                    fi
                    tmux has-session -t=$tmux_name 2>/dev/null
                    if [ $? = 0 ]; then
                      tmux_name="$tmux_name-1"
                    fi

                    tmux new-session -d -s $tmux_name
                    tmux send-keys -t "$tmux_name" "$setup_cmd; cd $snakemake_folder; bash $snakemake_cmd" Enter
                fi
                i_tmux=$((i_tmux+1))
            fi
        done
    fi
done

echo "Running tmux sessions:"
tmux ls
