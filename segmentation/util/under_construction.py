import numpy as np
import os
from collections import defaultdict

def create_2d_masks_gt():
    # creates separate 2d masks of the annotated ground truth
    path = r'C:\Users\niklas.khoss\Desktop\cp_vol\one_volume_seg.npy'
    arr = np.load(path, allow_pickle=True).item()
    masks = arr['masks']

    sv_path = r'C:\Users\niklas.khoss\Desktop\cp_vol\gt_masks_npy'
    c = 0
    for m in masks[1:]:
        np.save(os.path.join(sv_path, 'gt_mask_' + str(c)), m, allow_pickle=True)
        print(os.path.join(sv_path, 'gt_mask_' + str(c)))
        c += 1

# create_2d_masks_gt()

# Test area for 3d overlaps
def create_3d_match_dict(dataset1, dataset2):
    """

    Parameters
    ----------
    dataset1:   numpy array
        3d-numpy array (although 2d should work too).
    dataset2:   numpy array
        3d-numpy array (although 2d should work too).

    Returns
    -------
    matches_dict: dict
        Dictionary with: keys = neuron ID of dataset1, values = matched neuron IDs of dataset2.
    matches_area: dict
        Dictionary with: keys = neuron ID of dataset1, values = areas of matched neurons of dataset2.

    """

    print(f'ID matching start')

    # list of neuron IDs of dataset 1. remove '0'
    dataset1_uniq = np.unique(dataset1).astype(int)
    dataset1_uniq = dataset1_uniq[dataset1_uniq > 0]

    interim_match = list()
    interim_area = list()
    matches_dict = dict.fromkeys(dataset1_uniq)
    areas_dict = dict.fromkeys(dataset1_uniq)



    for neuron in dataset1_uniq:
        # loop over GT slices and check for neuron
        if neuron == 0:
            continue
        print(f'Neuron: {neuron}')

        # create 3D mask of neuron
        this_mask = dataset1 == neuron

        # check same slice of dataset2 and save all unique IDs, if 'this_mask' has matched anything
        match_mask = this_mask * dataset2

        # create list of IDs of overlapping neurons
        overlap_ids = list(np.unique(match_mask))
        overlap_ids = overlap_ids[overlap_ids > 0]

        for id in overlap_ids:
            # iterate over neurons and count the overlapping pixels
            if id == 0:
                continue

            this_overlap_mask = match_mask == id
            this_overlap_sum = np.count_nonzero(this_overlap_mask)

            # TODO add a threshold (pixels)!
            # if there is a match, non-zero values should
            if this_overlap_sum > 10:
                interim_match.append(id.astype(int))       # add IDs
                interim_area.append(this_overlap_sum)   # add matched area size

        # add ids and areas to dicts
        matches_dict[neuron] = interim_match
        areas_dict[neuron] = interim_area

        # clear interim
        interim_match = list()
        interim_area = list()

    return matches_dict, areas_dict

# Best overlap
def calc_best_overlap(mask_s0, # shall be 3d mask
                      masks_s1,
                      verbose=1):
    best_overlap = 0
    best_ind = None
    best_mask = np.zeros_like(masks_s1)
    all_vals = np.unique(masks_s1)

    for i, val in enumerate(all_vals):
        if val == 0:
            continue
        this_neuron_mask = masks_s1==val
        overlap = np.count_nonzero(mask_s0*this_neuron_mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_ind = i
            best_mask = this_neuron_mask
    if verbose >= 2:
        print(f'Best Neuron: {best_ind}, overlap of {best_overlap} between it and original')

    return best_overlap, best_mask

# refactor into using 3d arrays instead of 2d

