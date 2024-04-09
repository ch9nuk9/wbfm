#! /usr/bin/env bash

# Get the experimenter name from the command line
if [[ "$#" -eq 1 ]]; then
  EXPERIMENTER=${1%/}
else
  echo "Please rerun and pass the experimenter name"
  exit 1
fi

# Hardcoded example locations
PROJECT_DIR="/lisc/scratch/neurobiology/zimmer/Charles/dlc_stacks/"
RED_PATH="/lisc/scratch/neurobiology/ulises/wbfm/20211217/data/worm9/2021-12-17_18-34-25_worm9-channel-0-pco_camera1/2021-12-17_18-34-25_worm9-channel-0-pco_camera1bigtiff.btf"
GREEN_PATH="/lisc/scratch/neurobiology/ulises/wbfm/20211217/data/worm9/2021-12-17_18-34-25_worm9-channel-1-pco_camera2/2021-12-17_18-34-25_worm9-channel-1-pco_camera2bigtiff.btf"

# Actually run
COMMAND="scripts/0a-create_new_project.py"
python $COMMAND with project_dir=$PROJECT_DIR red_bigtiff_fname=$RED_PATH green_bigtiff_fname=$GREEN_PATH experimenter=$EXPERIMENTER

# Alternative: can just give the overall data path
#PARENT_DATA_FOLDER="/lisc/scratch/neurobiology/zimmer/ulises/wbfm/20220729/20220729_12ms/data/ZIM2156_worm12"
#
#python $COMMAND with project_dir=$PROJECT_DIR parent_data_folder=PARENT_DATA_FOLDER experimenter=$EXPERIMENTER
