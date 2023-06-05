#!/usr/bin/env bash

# This script is an example of how to use the 7z command to zip a file
# In general the files to zip will be in the /scratch directory, and the output_zip_path
# will be in the /archive directory. This is because the /scratch directory is
# temporary storage, and the /archive directory is permanent storage.
#
# Example usage:
#   sbatch 7z.sbatch -t /path/to/zip -o /path/to/output_zip_path

#SBATCH --job-name=zip   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=long             # Time limit hrs:min:sec
#SBATCH --output=zip.out    # Standard output and error log
#SBATCH --time=8:00:00
#SBATCH --mem=16G


pwd; hostname; date

while getopts t:o:n: flag
do
    case "${flag}" in
        t) target_files_to_zip=${OPTARG};;
        o) output_zip_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Actual command (could execute on a terminal)
7z a -t7z "$output_zip_path" "$target_files_to_zip"

date
echo "Finished; check logs to determine success"