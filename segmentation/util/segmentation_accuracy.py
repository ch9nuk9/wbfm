import numpy as np
import os
from segmentation.util.overlap import convert_to_3d, calc_all_overlaps, calc_best_overlap
from natsort import natsorted
import matplotlib.pyplot as plt


def seg_accuracy(ground_truth_path=None, algorithm_path=None):
    """
    Finds matching mask pairs and returns dictionaries containing the IDs of the neuron masks, which matched.
    Input needs to be in 3D format and already stitched (i.e. overlapping neurons in Z have the same ID).
    Dict entry is empty/None, when no match is found.

    Parameters
    ----------
    ground_truth_path: str
        Path to the ground truth data
    algorithm_path:    str
        Path to algorithm data

    Returns
    -------
    gt_to_algo: dict
        dictionary containing the matches of gt-masks to algo.
    algo_to_gt: dict
        dictionary containing the matches of algo-masks to gt.

    """
    # TODO rewrite the input to use the actual 3D-stitched arrays instead of file paths!

    # load stitched 3d arrays
    data_path = r'C:\Segmentation_working_area\stitched_3d_data'
    gt_3d = np.load(r'C:\Segmentation_working_area\stitched_3d_data\gt_stitched_3d.npy')
    algo_3d = np.load(r'C:\Segmentation_working_area\stitched_3d_data\stardist_fluo_stitched_3d.npy')

    # create 2 dictionaries (key = neuron ID, value = matched neuron ID):
    #   1. gt → algo
    #   2. algo → gt
    gt_to_algo, gt_to_algo_areas = create_3d_match_dict(gt_3d, algo_3d)
    algo_to_gt, algo_to_gt_areas = create_3d_match_dict(algo_3d, gt_3d)

    # false negatives = if neuron was not found by algorithm, i.e. [] in gt_to_algo
    # count empty in gt_to_algo
    fn_count = 0
    gt_vals = gt_to_algo.values()
    for v in gt_vals:
        if not v:
            fn_count += 1

    print(f'False negatives: {fn_count} of {len(gt_to_algo.keys())} or {round(fn_count / len(gt_to_algo.keys()), 3)}')

    # false positives = if neuron was found by algorithm, but not existent in ground truth, i.e. [] in algo_to_gt
    fp_count = 0
    algo_vals = algo_to_gt.values()
    for x in algo_vals:
        if not x:
            fp_count += 1

    print(f'False positives: {fp_count} of {len(algo_to_gt.keys())} or {round(fp_count / len(algo_to_gt.keys()), 3)}')

    # Oversegmentation: when the algorithm splits a neuron into 2 or more parts
    # i.e. when there is > 1 value for an entry when using GT as base for comparison with algo
    overseg = 0
    for o in gt_vals:
        if len(o) > 1:
            overseg += 1

    print(f'There are {overseg} instances of oversegmentation')

    # Undersegmentation: when algorithm fuses 2 neurons
    underseg = 0
    for u in algo_vals:
        if len(u) > 1:
            underseg += 1

    print(f'There are {underseg} instances of undersegmentation')

    # TODO save results in text file in algorithm folder

    return gt_to_algo, algo_to_gt

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
        overlap_ids = np.unique(match_mask)
        overlap_ids = list(overlap_ids[overlap_ids > 0])

        for id in overlap_ids:
            # iterate over neurons and count the overlapping pixels
            if id == 0:
                continue

            this_overlap_mask = match_mask == id
            this_overlap_sum = np.count_nonzero(this_overlap_mask)

            # if there is a match, non-zero values should
            interim_match.append(id.astype(int))       # add IDs
            interim_area.append(this_overlap_sum)   # add matched area size

        # add ids and areas to dicts
        matches_dict[neuron] = interim_match
        areas_dict[neuron] = interim_area

        # clear interim
        interim_match = list()
        interim_area = list()

    return matches_dict, areas_dict

# gt_3d = np.load(r'C:\Segmentation_working_area\stitched_3d_data\gt_stitched.npy')
# algo_3d = np.load(r'C:\Segmentation_working_area\stitched_3d_data\stardist_fluo_stitched.npy')
#
# matches = create_match_dict(gt_3d, algo_3d)
# seg_accuracy()
