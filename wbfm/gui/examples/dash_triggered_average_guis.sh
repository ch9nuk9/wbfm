#!/usr/bin/env bash

# This script opens several guis using the /home/charles/Current_work/repos/dlc_for_wbfm/wbfm/gui/interactive_two_dataframe_gui.py script.
# Example usage of the specific python script is:
# python interactive_two_dataframe_gui.py -p /path/to/folder -x 'body_segment_argmax' -y 'corr_max' -c 'genotype'

COMMAND="/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/gui/interactive_two_dataframe_gui.py"

# Each folder is in this subfolder, and in general will have different options for the x and y axes, and color.
PARENT_FOLDER="/home/charles/Current_work/presentations/Feb_2023"

# Define a function that does the following:
#   Open a tmux session, activate conda, and run the command with the options
function open_tmux_and_run() {
  # Args should be in the order: SUBFOLDER, X, Y, C, PORT
  # Session name should be unique, and include the port number
  SESS="dash_triggered_average_guis_${5}"
  tmux new-session -d -s $SESS
  tmux send-keys "conda activate wbfm38" C-m
  tmux send-keys "cd ${PARENT_FOLDER}/${1}" C-m
  tmux send-keys "python ${COMMAND} -p . -x '${2}' -y '${3}' -c ${4} --port ${5} --allow_public_access True" C-m
  echo "Opened ${PARENT_FOLDER}/${1} with port ${5}"
  echo "Accessible from the intranet at zimmer-ws00.neuro.univie.ac.at:${5}"
}

# Newest gui: semi-plateau
SUBFOLDER="volcano_semi_plateau-reversal_triggered"
X="effect size"
Y="-log(p value)"
C="genotype"
PORT="8050"
open_tmux_and_run ${SUBFOLDER} "${X}" "${Y}" ${C} ${PORT}

# Speed gui
SUBFOLDER="gui_speed_encodings"
X="genotype"
Y="multi_neuron"
C="genotype"
PORT="8051"
open_tmux_and_run ${SUBFOLDER} "${X}" "${Y}" ${C} ${PORT}