#!/bin/bash

# This script runs the Bayesian model for all neurons in the dataset

# Get an argument for whether to run gfp or not
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <gfp>"
  exit 1
fi

# Set the gfp flag
do_gfp=$1
if [ "$do_gfp" != "True" ] && [ "$do_gfp" != "False" ]; then
  echo "Invalid gfp flag: $do_gfp"
  exit 1
else
  echo "Running model with gfp: $do_gfp"
fi

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
'IL11LL'
'RIBR'
'AIBR'
'VB01'
'AIBL'
'IL2VL'
'URAVL'
'URAVR'
'IL2DR'
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

CMD="/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/utils/external/utils_pymc.py"
# Changes if running on gfp
if [ "$do_gfp" == "True" ]; then
  LOG_DIR="/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/logs"
else
  LOG_DIR="/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/logs"
fi

for neuron in "${neuron_list[@]}"
do
  echo "Dispatching model for neuron $neuron"
  log_fname="log_$neuron.txt"
  sbatch --time=1-10:00:00 --mem=32G --cpus-per-task=6 --job-name=bayesian_"$neuron" --wrap="python $CMD --neuron_name $neuron --do_gfp $do_gfp > $LOG_DIR/$log_fname"
#  break
  sleep 0.1
done