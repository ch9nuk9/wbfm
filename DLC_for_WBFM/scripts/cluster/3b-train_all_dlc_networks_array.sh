#!/bin/sh
#SBATCH --job-name=dlc_array   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=dlc_array_%A-%a.out    # Standard output and error log
#SBATCH --array=1-5                # Array range
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gpus-per-task=1
# # SBATCH --exclusive  # Tensorflow can easily run out of GPU memory if other jobs are running
# #SBATCH --gpu-bind=single:1
# See: https://stackoverflow.com/questions/67091056/gpu-allocation-in-slurm-gres-vs-gpus-per-task-and-mpirun-vs-srun

while getopts t:n: flag
do
    case "${flag}" in
        t) target_directory=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# this script loops over folders in a tracking folder
# Assumes these individual folders are Deeplabcut projects

pwd; hostname; date

module load cudnn/7.6.5

dir_array=($target_directory/*/)

for sub_dir in "${dir_array[@]}"; do
  echo "Found sub folders $sub_dir"
done

# Set the number of runs that each SLURM task should do
PER_TASK=$(( ${#dir_array[@]} / 5 ))
echo "Assigning $PER_TASK runs per task"

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK +1))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK))
CMD="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts/cluster/train_single_dlc_network.py"

# Print the task and run range
echo "This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM"

# Run the loop of runs for this task.
for (( run=$START_NUM; run<=$END_NUM; run++ )); do
  echo "This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run"
  dlc_config="${dir_array[run-1]}config.yaml"
  echo "---------${dlc_config}----------"
  if [ "$is_dry_run" ]; then
    full_command="$CMD --dlc_config $dlc_config"
    echo "Dry run with command: $full_command"
  else
    python $CMD --dlc_config "$dlc_config"
  fi
done
