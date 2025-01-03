#!/bin/bash

# Convert all Jupyter notebooks in target folder (or below) to Python scripts
# See https://stackoverflow.com/questions/45802690/converting-all-files-in-folder-to-py-files/49297443
# One answer is just incorrect; current last answer is correct

# Hardcoded for the paper notebooks (general version below)
# For paper notebooks:
bash /home/charles/Current_work/repos/dlc_for_wbfm/wbfm/scripts/hardcoded_protocols/convert_all_jupyter_notebooks_to_py.sh /home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/paper

# Usage:
# ./convert_all_jupyter_notebooks_to_py.sh /path/to/target/folder
# General command:
# find "$1" -not -path '*/.*' -name "*.ipynb" -exec jupyter nbconvert --to script {} \;
