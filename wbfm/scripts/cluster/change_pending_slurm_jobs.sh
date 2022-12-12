#!/bin/bash

# Designed to change all slurm jobs to the himem partition, which is often empty when the basic partition is full

# Get all slurm IDs from my user
readarray -t MY_IDS < <(squeue -u fieseler -h -o '%i')

# Define new slurm settings
NEW_SETTINGS="MinMemoryNode=600000 partition=himem"

# Loop through jobs and update each

for ID in ${MY_IDS[@]}; do
  echo "Running: update jobid=$ID $NEW_SETTINGS"
  scontrol update jobid="$ID" "$NEW_SETTINGS"
done
