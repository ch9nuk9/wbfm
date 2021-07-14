#!/usr/bin/env bash

COMMAND="gui/utils/manual_annotation.py"
PROJECT="Y:/shared_projects/wbfm/dlc_stacks/Charlie-worm3-new-seg/project_config.yaml"
NAME="Itamar"

python $COMMAND --project_path $PROJECT --corrector_name $NAME
