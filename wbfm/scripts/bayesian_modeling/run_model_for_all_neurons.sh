#!/bin/bash

# This script runs the Bayesian model for all neurons in the dataset

# First define the list of neurons
neuron_list=(
'AVEL'
'RID'
'AVBL'
'RMDVL'
'URYVL'
'BAGL'
'AUAR'
'RMED'
'RMEL'
'ALA'
'RMEV'
'RMDVR'
'ABVR'
'URYDL'
'RMER'
'VB02'
'URADL'
'IL2LL'
'SMDVR'
'RIML'
'AVER'
'SMDDR'
'AVAL'
'RIVR'
'BAGR'
'RIS'
'RIBL'
'IL1LL'
'OLQVL'
'DA01'
'URYVR'
'SMDVL'
'URADR'
'IL2LR'
'SIADL'
'RIVL'
'URXL'
'SMDDL'
'VB03'
'AVAR'
'IL1LR'
'VA02'
'ANTIcorR'
'URYDR'
'SIAVL'
'AVBR'
'DB01'
'ANTIcorL'
'SIAVR'
'SIADR'
'OLQVR'
'VA01'
'RIMR'
'IL2'
'URXR'
'AUAL'
'OLQDL'
'AQR'
'DB02'
'VG_middle_FWD_ramp_R'
'VG_post_turning_L'
'VG_post_turning_R'
'VG_anter_FWD_no_curve_R'
'VG_post_FWD_L'
'VG_post_FWD_R'
'VG_middle_FWD_ramp_L'
'IL11LL'
'RIBR'
'AIBR'
'VB01'
'AIBL'
'IL2VL'
'VG_middle_ramping_L'
'VG_middle_ramping_R'
'URAVL'
'URAVR'
'IL2DR'
'VG_anter_FWD_no_curve_L'
'OLQDR'
'IL2V'
'IL2DL'
'IL1DL'
'DD01'
'IL1VL'
'IL1VR'
'IL1DR'
'DA02'
)

# Now loop through the list of neurons and run the model
# But parallelize so that 12 are running at a time

CMD="/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/utils/external/utils_pymc.py"
LOG_DIR="/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/logs"

for neuron in "${neuron_list[@]}"
do
  echo "Running model for neuron $neuron"
  log_fname="log_$neuron.txt"
  python $CMD "$neuron" > "$LOG_DIR/$log_fname" &
  # DEBUG: just break after one run
#  break
  sleep 1
  while [ "$(jobs | wc -l)" -ge 12 ]; do
    sleep 10
  done
done