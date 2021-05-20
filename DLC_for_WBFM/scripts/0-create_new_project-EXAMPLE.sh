#!bash

RED_PATH="D:\More-stabilized-wbfm\test2020-10-22_16-15-20_test4-channel-0-pco_camera1\test2020-10-22_16-15-20_test4-channel-0-pco_camera1bigtiff.btf"
GREEN_PATH="D:\More-stabilized-wbfm\test2020-10-22_16-15-20_test4-channel-1-pco_camera2\test2020-10-22_16-15-20_test4-channel-1-pco_camera2bigtiff.btf"
COMMAND="scripts/create_new_project.py"

python $COMMAND with project_dir=./scratch red_bigtiff_fname=$RED_PATH green_bigtiff_fname=$GREEN_PATH
