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
'URYDL'
'RMER'
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
'URYVR'
'SMDVL'
'URADR'
'IL2LR'
'SIADL'
'RIVL'
'URXL'
'SMDDL'
'AVAR'
'IL1LR'
'ANTIcorR'
'URYDR'
'SIAVL'
'AVBR'
'ANTIcorL'
'SIAVR'
'SIADR'
'OLQVR'
'RIMR'
'IL2'
'URXR'
'AUAL'
'OLQDL'
'AQR'
'RIBR'
'AIBR'
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
'VA01'
'VA02'
'VB01'
'VB02'
'VB03'
'DA01'
'DA02'
'DB01'
'DB02'
'DD01'
'VG_anter_FWD_no_curve_L'
'VG_anter_FWD_no_curve_R'
'VG_middle_FWD_ramp_L'
'VG_middle_FWD_ramp_R'
'VG_middle_ramping_L'
'VG_middle_ramping_R'
'VG_post_FWD_L'
'VG_post_FWD_R'
'VG_post_turning_L'
'VG_post_turning_R'
)

# Now loop through the list of neurons and run the model
# But parallelize so that 12 are running at a time

CMD="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/utils/external/utils_pymc.py"
# Changes if running on gfp
if [ "$do_gfp" == "True" ]; then
  LOG_DIR="/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/logs"
else
  LOG_DIR="/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/logs"
fi

# I don't have access to the SLURM_ARRAY_TASK_ID variable, so I'm going to use the following workaround
# Create a temporary file to actually dispatch
# Create a temporary SLURM script
SLURM_SCRIPT=$(mktemp /tmp/slurm_script.XXXXXX)

# Write the SLURM script to handle array jobs
cat << EOF > $SLURM_SCRIPT
#!/bin/bash
#SBATCH --array=0-$(($NUM_TASKS-1))
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6

# Reproduce the list for the subfile
my_list=(${neuron_list[@]})
task_string=\${my_list[\$SLURM_ARRAY_TASK_ID]}
echo "Running model for neuron: \$task_string"
python $CMD --neuron_name \$task_string --do_gfp $do_gfp > $LOG_DIR/log_\$task_string.txt
EOF

# Submit the SLURM script
sbatch $SLURM_SCRIPT

# Clean up the temporary SLURM script
rm $SLURM_SCRIPT

# Dispatch an array job, with the index referring to the neuron
#num_jobs=${#neuron_list[@]}
#echo "Dispatching $num_jobs jobs"
#sbatch --array=0-$((num_jobs-1)) --time=1-00:00:00 --mem=32G --cpus-per-task=6 --wrap="python $CMD --neuron_name ${neuron_list[\$SLURM_ARRAY_TASK_ID]} --do_gfp $do_gfp > $LOG_DIR/log_${neuron_list[\$SLURM_ARRAY_TASK_ID]}.txt"
