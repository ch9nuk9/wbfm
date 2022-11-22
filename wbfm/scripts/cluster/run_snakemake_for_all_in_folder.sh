#!/bin/bash

# Get all user flags
while getopts t:n:s: flag
do
    case "${flag}" in
        t) folder_of_projects=${OPTARG};;
        n) is_dry_run=${OPTARG};;
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
                if [ "$is_dry_run" ]; then
                    # Run the snakemake dryrun
                    tmux_name="worm$i_tmux"
                    echo "DRYRUN: Dispatching on config file: $f_config with tmux name $tmux_name"
                else
                    # Get the snakemake command and run it
                    echo "TODO"
                fi
                i_tmux+=1
            fi
        done
    fi
done
