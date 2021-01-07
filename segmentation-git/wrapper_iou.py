"""
This is the wrapper function for calculating the IoUs.
It calls 'utils_iou.calculate_iou' and its inputs are the paths to:
    - ground truth data
    - cellpose results data (containing all parameter test subfolders)


Usage: python -m wrapper_iou.py gt_path cp_path

"""

import os
import sys
from utils_iou import calculate_iou

# ground truth path
ground_truth_path = r'/groups/zimmer/shared_projects/wbfm/ground_truth/one_volume_seg.npy'

# use input argument from bash script
cellpose_path = sys.argv[1]

# get current folder
#current_path = os.getcwd()

#

# pass gt_path and cp_path as arguments to calc_iou
calculate_iou(ground_truth_path, cellpose_path)

print('done with the wrapper')