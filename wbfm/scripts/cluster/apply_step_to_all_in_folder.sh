#!/usr/bin/env bash
# Example usage:
# bash apply_step_to_all_in_folder.sh -t '/scratch/neurobiology/zimmer/Charles/dlc_stacks/exposure_12ms' -s 0b
# Found within the package in:
# wbfm/scripts/cluster

# Get all user flags
while getopts t:n:s: flag
do
    case "${flag}" in
        t) folder_of_projects=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        s) step_reference=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Loop through the parent folder, then try to get the config file within each of these parent folders
for f in "$folder_of_projects"/*; do
    if [ -d "$f" ] && [ ! -L "$f" ]; then
        echo "Checking folder: $f"

        for f_config in "$f"/*; do
            if [ -f "$f_config" ] && [ "${f_config##*/}" = "project_config.yaml" ]; then
                if [ "$is_dry_run" ]; then
                    echo "DRYRUN: Dispatching on config file: $f_config"
                else
                    sbatch ./single_step_dispatcher.sbatch -s "$step_reference" -t "$f_config"
                fi
            fi
        done


    fi
done