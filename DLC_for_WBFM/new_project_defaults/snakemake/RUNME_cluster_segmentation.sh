#!/bin/bash

# Example from: https://hackmd.io/@bluegenes/BJPrrj7WB

OPT="sbatch -t {cluster.time} -p {cluster.partition} --cpus-per-task {cluster.cpus_per_task} --mem {cluster.mem} --output {cluster.output}"
NUM_JOBS_TO_SUBMIT=2

# Needs writable cache
export HOME="/scratch/neurobiology/zimmer"

snakemake -s segmentation_pipeline.smk --latency-wait 60 --cluster "$OPT" --cluster-config cluster_config.yaml --jobs $NUM_JOBS_TO_SUBMIT
