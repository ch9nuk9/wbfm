#!/usr/bin/bash

#ENV_CMD="source activate DLC-for-WBFM;"
PYTHON_CMD="/users/charles.fieseler/wbfm/wbfm/scripts/visualization/0+copy_video_data.py"
PROJECT="project_path=/groups/zimmer/shared_projects/wbfm/dlc_stacks/Charlie-worm3-long/project_config.yaml"
OPT="tiff_not_zarr=False copy_locally=False"

LOG="/groups/zimmer/shared_projects/wbfm/dlc_stacks/Charlie-worm3-long/log.log"

sbatch --mem 128G --wrap="python $PYTHON_CMD with $PROJECT $OPT > $LOG"