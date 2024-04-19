#!/bin/bash

# Designed to change all slurm jobs to the himem partition, which is often empty when the basic partition is full

# Get user name
USER=$(whoami)

# Get all slurm IDs
readarray -t MY_IDS < <(squeue -u "$USER" -h -o '%i')

# Loop through jobs and update each
for ID in ${MY_IDS[@]}; do
  echo "If pending, moving job to himem partition: scontrol update jobid=$ID MinMemoryNode=600000 partition=himem"
  scontrol update jobid=$ID MinMemoryNode=600000 partition=himem
done
