#!/bin/bash

# This script runs the Bayesian model for all neurons in the dataset

# Function to display a help message
function show_help {
  echo "Usage: $0 [-g] [-r] [-h] <gfp>"
  echo "  -g: Use GFP data"
  echo "  -r: Trace mode; should be one of 'None', 'pca_global', 'pca_global_1'"
  echo "  -h: Show this help message"
}

# Get all user flags
use_gfp="false"
use_raw_trace="false"
while getopts gr: flag
do
    case "${flag}" in
        g) use_gfp="true";;
        r) trace_mode=${OPTARG};;
        h) show_help
           exit 0;;
        *) raise error "Unknown flag"
    esac
done

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
'SIAVL'
'SIAVR'
'SAAVL'
'SAAVR'
'SIADL'
'SIADR'
'RIAL'
'RIAR'
'RMDDL'
'RMDDR'
'RMDVL'
'RMDVR'
'AVFL'
'AVFR'
'AWBL'
'AWBR'
'AWAL'
'AWAR'
'IL1LL'
'IL1LR'
'IL2LL'
'IL2LR'
)

# Now loop through the list of neurons and run the model
# But parallelize so that 12 are running at a time

CMD="/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/utils/external/utils_pymc.py"
# Changes if running on gfp
if [ "$do_gfp" == "true" ]; then
  LOG_DIR="/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/logs"
else
  LOG_DIR="/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/logs"
fi

# I don't have access to the SLURM_ARRAY_TASK_ID variable, so I'm going to use the following workaround
# Create a temporary file to actually dispatch
SLURM_SCRIPT=$(mktemp /tmp/slurm_script.XXXXXX)
NUM_TASKS=${#neuron_list[@]}

# Set of option-specific variables
# gfp datasets are much faster to run
if [ "$use_gfp" == "true" ]; then
  CMD="$CMD --do_gfp"
  NUM_HOURS=6
else
  NUM_HOURS=18
fi

if [ "$trace_mode" ]; then
  CMD="$CMD --use_raw_trace $trace_mode"
fi

# Actually run
cat << EOF > $SLURM_SCRIPT
#!/bin/bash
#SBATCH --array=0-$(($NUM_TASKS-1))
#SBATCH --time=0-0$NUM_HOURS:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --partition=short,basic

# Reproduce the list for the subfile
my_list=(${neuron_list[@]})
task_string=\${my_list[\$SLURM_ARRAY_TASK_ID]}
echo "Running model for neuron: \$task_string with command: $CMD"
python $CMD --neuron_name \$task_string > $LOG_DIR/log_\$task_string.txt
EOF

# Submit the SLURM script
sbatch $SLURM_SCRIPT

# Clean up the temporary SLURM script
rm $SLURM_SCRIPT
