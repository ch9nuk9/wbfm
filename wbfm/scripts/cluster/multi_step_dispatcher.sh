#!/usr/bin/env bash

# Pass command steps in order, without a tag

# Get all user flags
while getopts t:n:s: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        s) step_reference_array+=("$OPTARG");;
        *) raise error "Unknown flag"
    esac
done

for step_reference in ${step_reference_array[@]}; do
  sbatch --wait single_step_dispatcher.sbatch -s "$step_reference" -t "$project_path"
done
