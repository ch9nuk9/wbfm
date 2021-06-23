#!/usr/bin/bash

ENV_CMD="conda activate DLC-for-WBFM;"
PYTHON_CMD="/users/charles.fieseler/DLC_for_WBFM/DLC_for_WBFM/scripts/visualization/0+copy_video_data.py"
PROJECT="project_path=/groups/zimmer/shared_projects/wbfm/dlc_stacks/Charlie-immobilized-long/project_config.yaml"
OPT="tiff_not_zarr=False copy_locally=False DEBUG=True"

srun "$ENV_CMD python $PYTHON_CMD with $PROJECT $OPT"