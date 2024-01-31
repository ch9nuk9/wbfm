#!/bin/bash

# Add help function
function usage {
    echo "Usage: $0 [-s rule] [-n] [-c] [-h]"
    echo "  -s: snakemake rule to run (default: traces_and_behavior; other options: traces, behavior)"
    echo "  -n: dry run (default: false)"
    echo "  -c: do NOT use cluster (default: false, i.e. run on cluster)"
    echo "  -h: display help (this message)"
    exit 1
}

# Example from: https://hackmd.io/@bluegenes/BJPrrj7WB

# Get the snakemake rule to run as a command line arg

# Get other command line args
# Set defaults: not a dry run and on the cluster
DRYRUN=""
USE_CLUSTER="True"
RULE="traces_and_behavior"

while getopts :s:nch flag
do
    case "${flag}" in
        s) RULE=${OPTARG};;
        n) DRYRUN="True";;
        c) USE_CLUSTER="";;
        h) usage;;
        *) echo "Unknown flag"; usage; exit 1;;
    esac
done

# Package slurm options
OPT="sbatch -t {cluster.time} -p {cluster.partition} --cpus-per-task {cluster.cpus_per_task} --mem {cluster.mem} --output {cluster.output} --gres {cluster.gres}"
NUM_JOBS_TO_SUBMIT=4

# Actual command
if [ "$DRYRUN" ]; then
    echo "DRYRUN of snakemake rule: $RULE. Common options: traces_and_behavior (default), traces, behavior"
    snakemake "$RULE" --debug-dag -n -s pipeline.smk --cores
elif [ -z "$USE_CLUSTER" ]; then
    echo "Running snakemake rule locally: $RULE. Common options: traces_and_behavior (default), traces, behavior"
    snakemake "$RULE" -s pipeline.smk --latency-wait 60 --cores 56
else
    echo "Running snakemake rule on the cluster: $RULE. Common options: traces_and_behavior (default), traces, behavior"
    snakemake "$RULE" -s pipeline.smk --latency-wait 60 --cluster "$OPT" --cluster-config cluster_config.yaml --jobs $NUM_JOBS_TO_SUBMIT
fi
