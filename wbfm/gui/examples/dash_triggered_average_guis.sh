#!/usr/bin/env bash

# This script opens several guis using the /home/charles/Current_work/repos/dlc_for_wbfm/wbfm/gui/interactive_two_dataframe_gui.py script.
# Example usage of the specific python script is:
# python interactive_two_dataframe_gui.py -p /path/to/folder -x 'body_segment_argmax' -y 'corr_max' -c 'genotype'

COMMAND="/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/gui/interactive_two_dataframe_gui.py"

# Each folder is in this subfolder, and in general will have different options for the x and y axes, and color.
PARENT_FOLDER="/home/charles/Current_work/presentations/Feb_2023"

# Define a list of options for each call: folder, x, y, and c
SUBFOLDER="volcano_semi_plateau-reversal_triggered"
X="effect size"
Y="-log(p value)"
C="genotype"
PORT="8050"

# Open a tmux session, activate conda, and run the command with the options
tmux new-session -d -s "dash_triggered_average_guis"
tmux send-keys "conda activate wbfm38" C-m
tmux send-keys "cd ${PARENT_FOLDER}/${SUBFOLDER}" C-m
tmux send-keys "python ${COMMAND} -p . -x '${X}' -y '${Y}' -c ${C} --port ${PORT} --allow_public_access True" C-m
echo "Opened ${PARENT_FOLDER}/${SUBFOLDER} with port ${PORT}"
echo "Accessible from the intranet at zimmer-ws00.neuro.univie.ac.at:${PORT}"
