#!/usr/bin/env bash

# Pass command steps in order.
# Note: this bash command must run until everything finishes, so tmux is suggested
#
# Example 1 (changing the preprocessed image, and recalculating traces):
# bash multi_step_dispatcher.sh -s 0b -s 4-alt -t '/scratch/neurobiology/zimmer/Charles/dlc_stacks/gcamp7b/ZIM2165_Gcamp7b_worm8_immobilised-2022_10_14_sharpened_1_10/project_config.yaml'
#
# Example 2 (use manual annotation or somehow modified tracklets to rebuild segmentation, build tracking, and recalculate traces):
# bash multi_step_dispatcher.sh -s 1-alt -s 3b -s 4 -t '/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28/project_config.yaml'


# Get all user flags
while getopts t:n:s: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        s) step_reference_array+=("$OPTARG");;
        *) raise error "Unknown flag"
    esac
done

NUM_JOBS=$(( ( $# / 2 ) - 1 ))
echo "Found $NUM_JOBS jobs"

for step_reference in "${step_reference_array[@]}"; do
  sbatch --wait single_step_dispatcher.sbatch -s "$step_reference" -t "$project_path"
done
