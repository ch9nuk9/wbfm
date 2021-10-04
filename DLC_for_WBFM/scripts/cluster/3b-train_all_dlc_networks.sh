#!/bin/bash

TARGET_DIR="$1"
echo "Running on parent directory: $TARGET_DIR"

# TODO: actually use an array job, not just looping
i=0
for f in $TARGET_DIR; do
    if [ -d "$f" ]; then
        dlc_config="${f}config.yaml"

        sbatch --job-name=dlc_$i --output=dlc_$i.out train_single_dlc_network.sbatch $dlc_config
        i=$((i+1))
    fi
done
