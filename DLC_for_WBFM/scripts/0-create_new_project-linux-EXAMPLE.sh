#!bash

# Get the experimenter name from the command line
if [[ "$#" -eq 1 ]]; then
  EXPERIMENTER=${1%/}
else
  echo "Please rerun and pass the experimenter name"
  exit 1
fi

# Hardcoded example locations
PROJECT_DIR="/groups/zimmer/shared_projects/wbfm/tutorial"
RED_PATH="/groups/zimmer/shared_projects/wbfm/dat/immobilized_with_WBFM_hardware/ZIM2051_trial_21_HEAD_mcherry_FULL_bigtiff.btf"
GREEN_PATH="/groups/zimmer/shared_projects/wbfm/dat/immobilized_with_WBFM_hardware/ZIM2051_trial_21_HEAD_gcamp_FULL_bigtiff.btf"

# Actually run
COMMAND="scripts/0-create_new_project.py"
python $COMMAND with project_dir=$PROJECT_DIR red_bigtiff_fname=$RED_PATH green_bigtiff_fname=$GREEN_PATH experimenter=EXPERIMENTER
