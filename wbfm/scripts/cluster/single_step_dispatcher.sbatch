#!/usr/bin/env bash

#SBATCH --job-name=single_step
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=medium             # Time limit hrs:min:sec
#SBATCH --output=log_single_step-%j.out    # Standard output and error log
#SBATCH --time=12:00:00  # A normal video should only take <3 hours, but some could be longer
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=himem

pwd; hostname; date

# Necessary because the default home directory is not writable from the cluster
# UPDATE: as of August 2022 this is not necessary
#export HOME="/scratch/neurobiology/zimmer/Charles"

# Get all user flags
while getopts t:n:s: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        s) step_reference=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Unpack the flag into an actual command
CMD_DIR="/scratch/neurobiology/zimmer/Charles/github_repos/dlc_for_wbfm/wbfm/scripts"

source "$(dirname "$0")"/utils/dict_of_sbatch_commands.sh

CMD="${CMD_DIR}/${CMD_DICT[$step_reference]}"
FULL_CMD="python $CMD with project_path=$project_path"

echo "Running step $step_reference using full command:"
echo "$FULL_CMD"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $FULL_CMD"
else
  python "$CMD" with project_path="$project_path"
fi

date
echo "Finished; check logs to determine success"
