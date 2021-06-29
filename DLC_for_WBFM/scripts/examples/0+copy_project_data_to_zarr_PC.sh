#!/usr/bin/bash

#ENV_CMD="source activate DLC-for-WBFM;"
PYTHON_CMD="C:/Users/charles.fieseler/Documents/Current_work/DLC_for_WBFM/DLC_for_WBFM/scripts/visualization/0+copy_video_data.py"
PROJECT="project_path=Y:/shared_projects/wbfm/dlc_stacks/Charlie-worm3-long/project_config.yaml"
OPT="tiff_not_zarr=False copy_locally=False"

LOG="Y:/shared_projects/wbfm/dlc_stacks/Charlie-worm3-long/log.log"

python $PYTHON_CMD with $PROJECT $OPT > $LOG