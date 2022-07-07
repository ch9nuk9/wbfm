#!/usr/bin/env bash

# Get the experimenter name from the command line
if [[ "$#" -eq 1 ]]; then
  EXPERIMENTER=${1%/}
else
  echo "Please rerun and pass the experimenter name"
  exit 1
fi

# Hardcoded example locations
#PROJECT_DIR="Y:/shared_projects/wbfm/tutorial"
#RED_PATH="Y:/shared_projects/wbfm/dat/immobilized_with_WBFM_hardware/ZIM2051_trial_21_HEAD_mcherry_FULL_bigtiff.btf"
#GREEN_PATH="Y:/shared_projects/wbfm/dat/immobilized_with_WBFM_hardware/ZIM2051_trial_21_HEAD_gcamp_FULL_bigtiff.btf"

# Hardcoded example locations
PROJECT_DIR="Y:/shared_projects/wbfm/dlc_stacks"
RED_PATH="Y:/shared_projects/wbfm/dat/ZIM2051/2021-03-04_16-07-57_worm3_ZIM2051/2021-03-04_16-07-57_worm3_ZIM2051-channel-0-pco_camera1/2021-03-04_16-07-57_worm3_ZIM2051-channel-0-pco_camera1bigtiff.btf"
GREEN_PATH="Y:/shared_projects/wbfm/dat/ZIM2051/2021-03-04_16-07-57_worm3_ZIM2051/2021-03-04_16-07-57_worm3_ZIM2051-channel-1-pco_camera2/2021-03-04_16-07-57_worm3_ZIM2051-channel-1-pco_camera2bigtiff.btf"

# Actually run
COMMAND="scripts/0a-create_new_project.py"
python $COMMAND with project_dir=$PROJECT_DIR red_bigtiff_fname=$RED_PATH green_bigtiff_fname=$GREEN_PATH experimenter=$EXPERIMENTER
