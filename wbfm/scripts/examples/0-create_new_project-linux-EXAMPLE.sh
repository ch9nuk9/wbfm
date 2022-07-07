#! /usr/bin/env bash

# Get the experimenter name from the command line
if [[ "$#" -eq 1 ]]; then
  EXPERIMENTER=${1%/}
else
  echo "Please rerun and pass the experimenter name"
  exit 1
fi

# Hardcoded example locations
#PROJECT_DIR="/scratch/zimmer/Charles/dlc_stacks/"
#RED_PATH="/project/neurobiology/zimmer/wbfm/dat/ZIM2051/2021-03-04_16-07-57_worm3_ZIM2051/2021-03-04_16-07-57_worm3_ZIM2051-channel-0-pco_camera1/2021-03-04_16-07-57_worm3_ZIM2051-channel-0-pco_camera1.btf"
#GREEN_PATH="/project/neurobiology/zimmer/wbfm/dat/ZIM2051/2021-03-04_16-07-57_worm3_ZIM2051/2021-03-04_16-07-57_worm3_ZIM2051-channel-0-pco_camera1/2021-03-04_16-07-57_worm3_ZIM2051-channel-0-pco_camera1.btf"

PROJECT_DIR="/scratch/zimmer/Charles/dlc_stacks/"
RED_PATH="/scratch/ulises/wbfm/20211217/data/worm9/2021-12-17_18-34-25_worm9-channel-0-pco_camera1/2021-12-17_18-34-25_worm9-channel-0-pco_camera1bigtiff.btf"
GREEN_PATH="/scratch/ulises/wbfm/20211217/data/worm9/2021-12-17_18-34-25_worm9-channel-1-pco_camera2/2021-12-17_18-34-25_worm9-channel-1-pco_camera2bigtiff.btf"

# Actually run
COMMAND="scripts/0a-create_new_project.py"
python $COMMAND with project_dir=$PROJECT_DIR red_bigtiff_fname=$RED_PATH green_bigtiff_fname=$GREEN_PATH experimenter=$EXPERIMENTER
