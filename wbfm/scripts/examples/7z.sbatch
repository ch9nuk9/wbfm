#!/usr/bin/env bash

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
        t) path_to_zip=${OPTARG};;
        o) output=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

7z a -t7z $output $path_to_zip

date
echo "Finished; check logs to determine success"