#!/bin/bash

# Example from: https://hackmd.io/@bluegenes/BJPrrj7WB

OPT="sbatch -t {cluster.time} -p {cluster.partition} --cpus-per-task {cluster.cpus_per_task} --mem {cluster.mem} --output {cluster.output}"
NUM_JOBS_TO_SUBMIT=2

# Needs writable cache
# As of 8/2022 your home folder at /home/user should be writable from the cluster, but this may be temporary
# export HOME="/scratch/neurobiology/zimmer/YOUR/USER"

snakemake -s pipeline.smk --latency-wait 60 --cluster "$OPT" --cluster-config cluster_config.yaml --jobs $NUM_JOBS_TO_SUBMIT