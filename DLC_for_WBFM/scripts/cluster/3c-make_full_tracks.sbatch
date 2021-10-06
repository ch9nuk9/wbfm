#!/usr/bin/env bash

while getopts t:n: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

CMD="3c-make_full_tracks.sh"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $CMD $project_path"
else
  sbatch $CMD "$project_path"
fi
