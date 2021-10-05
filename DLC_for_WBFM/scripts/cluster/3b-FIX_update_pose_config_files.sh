#!/usr/bin/env bash

# Updates the config files for all dlc sub-projects to point to the correct cluster locations
# In detail: DLC uses pretrained resnet weights that are stored in the conda environment (hard coded)
# So if you create the project on windows, these paths won't be correct

# First, reset all the paths to work for the cluster
PROJECT="$1"
INIT_WEIGHTS="/home/user/fieseler/.conda/envs/wbfm-cluster/lib/python3.8/site-packages/deeplabcut/pose_estimation_tensorflow/models/pretrained/resnet_v1_50.ckpt"
CMD="/scratch/zimmer/Charles/github_repos/dlc_for_wbfm/DLC_for_WBFM/scripts/alternate/3b-alternate-update_all_pose_configs.py"
python $CMD with project_path=$PROJECT update_key=init_weights update_val=$INIT_WEIGHTS

# Then, reset all the paths that can start from my pretrained networks
python $CMD with project_path=$PROJECT

echo "Finished"
date
