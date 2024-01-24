#!/bin/bash

# Add help function
function usage {
    echo "Usage: $0 [-n] [-c] [-h] [rule]"
    echo "  -n: dry run"
    echo "  -c: use cluster"
    echo "  -h: help (this message)"
    echo "  rule: snakemake rule to run (default: traces_and_behavior)"
    exit 1
}

# Example from: https://hackmd.io/@bluegenes/BJPrrj7WB

# Get the snakemake rule to run as a command line arg
if [ $# -eq 0 ]
then
    RULE="traces_and_behavior"
else
    RULE=$1
fi
echo "Running snakemake rule: $RULE. Common options: traces_and_behavior (default), traces, behavior"

# Get other command line args
# Set defaults: not a dry run and on the cluster
DRYRUN=""
USE_CLUSTER="True"

while getopts n:c:h: flag
do
    case "${flag}" in
        n) DRYRUN=${OPTARG};;
        c) USE_CLUSTER=${OPTARG};;
        h) usage;;
        *) raise error "Unknown flag"
    esac
done

# Package slurm options
OPT="sbatch -t {cluster.time} -p {cluster.partition} --cpus-per-task {cluster.cpus_per_task} --mem {cluster.mem} --output {cluster.output} --gres {cluster.gres}"
NUM_JOBS_TO_SUBMIT=4

# Actual command
if [ "$DRYRUN" ]; then
    snakemake "$RULE" --debug-dag -n -s pipeline.smk --cores
elif [ "$USE_CLUSTER" ]; then
    snakemake "$RULE" -s pipeline.smk --latency-wait 60 --cores 56
else
    snakemake "$RULE" -s pipeline.smk --latency-wait 60 --cluster "$OPT" --cluster-config cluster_config.yaml --jobs $NUM_JOBS_TO_SUBMIT
fi
