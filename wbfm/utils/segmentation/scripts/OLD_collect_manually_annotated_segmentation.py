#!/usr/bin/env python
# coding: utf-8

import os
import zarr
import tifffile
import numpy as np
import pandas as pd


# Input format: Multiple zarr files for each type of data
# Stardist desired output: single tif file for everything
# Strategy: read all data and masks into numpy, then save


# Note: lives on the cluster
dir_name = "/home/project/neurobiology/zimmer/wbfm/Annotation_party"

all_data = []
all_masks = []

index = ['False positives', 'False negatives', 'Undersegmentation', 'Oversegmentation', 'Total']
df = pd.DataFrame(index=index)


def collect_masks_from_subfolder(project, stats, finished_ind):
    """Modifies all_data and all_masks in place"""

    mask_fname = os.path.join(dir_name, project, "1-segmentation", "corrected_masks.zarr")
    data_fname = os.path.join(dir_name, project, "data_subset.tiff")
    # Read data and masks
    dat = tifffile.imread(data_fname)
    masks = zarr.open_array(mask_fname)
    rows = map(lambda x: project + str(x), finished_ind)
    for i, (r, stat) in enumerate(zip(rows, stats)):
        stat.append(len(np.unique(masks[i, ...])) - 1)
        df[r] = stat
    all_data.extend([dat[i, ...] for i in finished_ind])
    all_masks.extend([masks[i, ...] for i in finished_ind])


# Collect individual datasets

## Itamar
project = "Party-GFP_normal-2021_05_28"
stats = [[4, 18, 1, 7],
         [5, 14, 0, 5]]
# Collect from google doc by hand
finished_ind = [0, 9]
collect_masks_from_subfolder(project, stats, finished_ind)

## Kerem

project = "Party-GFP_turning_3d-2021_05_28"
stats = [[4, 22, 0, 2],
         [0, 23, 0, 0]]
finished_ind = [0, 1]

collect_masks_from_subfolder(project, stats, finished_ind)


## Lukas

project = "Party-GFP_turning_2d-2021_05_28"
finished_ind = [0, 1, 2]
stats = [[0, 12, 4, 3],
         [1, 9, 2, 2]]

collect_masks_from_subfolder(project, stats, finished_ind)


## Nisa

project = "Party-Normal_normal-2021_05_28"
# Collect from google doc by hand
finished_ind = [0, 1, 2]

rows = map(lambda x: project + str(x), finished_ind)
stats = [[2, 6, 3, 1],
         [3, 7, 6, 2],
         [1, 20, 26, 2]]

collect_masks_from_subfolder(project, stats, finished_ind)


## Ulises

project = "Party-Normal_turning_3d-2021_05_28"
finished_ind = [0, 1, 2]
stats = [[1, 2, 1, 1],
         [3, 4, 1, 0],
         [1, 4, 0, 0]]

collect_masks_from_subfolder(project, stats, finished_ind)


## Rebecca

project = "Party-Normal_turning_2d-2021_05_28"
finished_ind = [0]
stats = [[0, 3, 10, 2]]

collect_masks_from_subfolder(project, stats, finished_ind)


# # Charlie - DIFFERENT SIZE

# In[10]:

# project = "Party-Immobilized-2021_05_28"

# mask_fname = os.path.join(dir_name, project, "1-segmentation", "corrected_masks.zarr")
# data_fname = os.path.join(dir_name, project, "data_subset.tiff")

# # Collect from google doc by hand
# finished_ind = [0]

# rows = map(lambda x: project + str(x), finished_ind)
# stats = [[0,23,1,1]]

# for r, stat in zip(rows, stats):
#     df[r] = stat

# # Read data and masks
# dat = tifffile.imread(data_fname)
# masks = zarr.open_array(mask_fname)

# all_data.extend([dat[i,...] for i in finished_ind])
# all_masks.extend([masks[i,...] for i in finished_ind])

# Save

out_fname = 'segmentation_party_statistics.pickle'
df.to_pickle(out_fname)

