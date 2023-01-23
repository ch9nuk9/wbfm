#!/usr/bin/env python
# coding: utf-8

import zarr
import os
import numpy as np
import napari


# This script collects the different subfolders of annotation and saves them in a stardist-ready format

# Get the several different annotations

base_dir = '/home/project/neurobiology/zimmer/wbfm/ground_truth_segmentation/student_helper_annotations'


def save_annotated_subset(annotated_times, data_path, seg_path, output_name_suffix, to_remove_flyback=False):
    # Note that this lets zarr automatically choose the chunk size
    dat = zarr.open(data_path)
    seg = zarr.open(seg_path)
    
    # List indexing doesn't work unless you specify all dims
    dat_subset = np.stack([dat[t] for t in annotated_times], axis=0)
    seg_subset = np.stack([seg[t] for t in annotated_times], axis=0)
    
    if to_remove_flyback:
        dat_subset = dat_subset[:, 1:, ...]
        seg_subset = seg_subset[:, 1:, ...]
    
    print(dat_subset.shape)
    fname = os.path.join(base_dir, f'red_{output_name_suffix}.zarr')
    zarr.save_array(fname, dat_subset)
    fname = os.path.join(base_dir, f'masks_{output_name_suffix}.zarr')
    zarr.save_array(fname, seg_subset)


def just_plot(annotated_times, data_path, seg_path, to_remove_flyback=False):
    dat = zarr.open(data_path)
    seg = zarr.open(seg_path)
    
    # List indexing doesn't work unless you specify all dims
    dat_subset = np.stack([dat[t] for t in annotated_times], axis=0)
    seg_subset = np.stack([seg[t] for t in annotated_times], axis=0)
    
    if to_remove_flyback:
        dat_subset = dat_subset[:, 1:, ...]
        seg_subset = seg_subset[:, 1:, ...]
    
    print(dat_subset.shape)
    
    v = napari.Viewer(ndisplay=3)
    v.add_image(dat_subset)
    v.add_labels(seg_subset)


# ### Freely moving

# worm 5
# From google sheet: https://docs.google.com/spreadsheets/d/1bR6J7_buxBA6bL7OD1OT4H_ZgrfR1W1NLxTRK5f-aJ0/edit#gid=1231122446
# annotated_times = [151, 378, 1662, 938,2348,2462,475,2462,3004,1012,1059,2356,2465,3013,3016,382,970,1286,1635,2397,392,1036,2977,2996,3232,1053,149]
# annotated_times = [151, 938, 2348, 475, 3004, 3016, 382, 970, 1286, 1635, 2397, 2977, 2996, 3232, 149]
#
# data_path = '/scratch/neurobiology/zimmer/Charles/dat_for_students/worm5/2022-01-27_21-49_worm5_Ch0bigtiff.zarr'
# seg_path = '/scratch/neurobiology/zimmer/Charles/dlc_stacks/students/students-worm5-resegment/1-segmentation/masks-update.zarr'
#
# output_name_suffix = 'worm_5'
#
# save_annotated_subset(annotated_times, data_path, seg_path, output_name_suffix, to_remove_flyback=True)

# Worm 4
# annotated_times = [15, 501,725,724,992,2028,2777]
annotated_times = [15, 501, 725, 724, 2028, 2777]

project_folder = "/home/project/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/"

data_path = os.path.join(project_folder, 'round1_worm4/dat/2022-01-27_21-26_worm4_Ch0bigtiff.btf_preprocessed.zarr.zip')
# TODO: is this actually the updated segmentation?
seg_path = os.path.join(project_folder, 'round1_worm4/1-segmentation/masks.zarr')
# seg_path = '/scratch/neurobiology/zimmer/Charles/dlc_stacks/students/students-worm4-resegment/1-segmentation/masks3d.zarr'

output_name_suffix = 'worm_4'

save_annotated_subset(annotated_times, data_path, seg_path, output_name_suffix, to_remove_flyback=True)

# ### Immobilized
# Worm 6
# annotated_times = [219,698,1259,1752,2307,2819]
# annotated_times = [219]
#
# data_path = '/scratch/neurobiology/zimmer/Charles/dat_for_students/immobilized_worm6/2022-02-28_15-36_ZIM2156_immobilised_worm_6_Ch0/2022-02-28_15-36_ZIM2156_immobilised_worm_6_Ch0bigtiff.zarr'
# seg_path = '/scratch/neurobiology/zimmer/Charles/dlc_stacks/students/immobilized_worm6/1-segmentation/masks.zarr'
#
# output_name_suffix = 'imm_worm_6'
# save_annotated_subset(annotated_times, data_path, seg_path, output_name_suffix)

# Worm 4
# annotated_times = [219,698,1259]
# annotated_times = [219, 698]
#
# data_path = '/scratch/neurobiology/zimmer/Charles/dat_for_students/immobilized_worm4/2022-02-28_14-14_ZIM2156_immobilised_worm_4_Ch0/2022-02-28_14-14_ZIM2156_immobilised_worm_4_Ch0bigtiff.zarr'
# seg_path = '/scratch/neurobiology/zimmer/Charles/dlc_stacks/students/immobilized_worm4/1-segmentation/masks.zarr'
#
# output_name_suffix = 'imm_worm_4'
#
# save_annotated_subset(annotated_times, data_path, seg_path, output_name_suffix)
