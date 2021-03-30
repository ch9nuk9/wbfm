import numpy as np
import pickle


def seg_accuracy(ground_truth_path=None, algorithm_path=None, verbose=0):
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
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    accuracy_results : pickle
        Pickle file containing metrics of accuracy calculations including false/true pos/neg, under/oversegmentations &
        volumes of matched neuron masks.

    """

    gt_3d = np.load(ground_truth_path)
    algo_3d = np.load(algorithm_path)

    # create 2 dictionaries (key = neuron ID, value = matched neuron ID):
    # gt_to_algo: dict
    #       dictionary contains: algorithm results tried to match GT-masks.
    # {GT neuron ID: list(algorithm match IDs)}

    gt_to_algo, gt_to_algo_volumes = create_3d_match_dict(gt_3d, algo_3d, verbose-1)
    gt_to_algo, gt_to_algo_volumes = remove_small_overlaps(gt_to_algo, gt_to_algo_volumes, gt_3d, verbose=verbose-1)

    algo_to_gt, algo_to_gt_volumes = create_3d_match_dict(algo_3d, gt_3d, verbose-1)
    algo_to_gt, algo_to_gt_volumes = remove_small_overlaps(algo_to_gt, algo_to_gt_volumes, algo_3d, verbose=verbose-1)

    # false negatives = if neuron was not found by algorithm, i.e. [] in gt_to_algo
    # count empty in gt_to_algo
    fn_count = 0
    algo_vals = gt_to_algo.values()
    for v in algo_vals:
        if not v:
            fn_count += 1

    fn_p = round(fn_count / len(gt_to_algo.keys()), 3) * 100

    # false positives = if neuron was found by algorithm, but not existent in ground truth, i.e. [] in algo_to_gt
    fp_count = 0
    gt_vals = algo_to_gt.values()
    for x in gt_vals:
        if not x:
            fp_count += 1

    fp_p = round(fp_count / len(gt_to_algo.keys()), 3) * 100

    # True positives = if existing neuron was found by algorithm
    tp_count = 0
    for k, v in gt_to_algo.items():
        if v:
            tp_count += 1
            # TODO check TPs

    tp_p = round(tp_count / len(gt_to_algo.keys()), 3) * 100

    # Oversegmentation: when the algorithm splits a neuron into 2 or more parts
    # i.e. when there is > 1 value for an entry when using GT as base for comparison with algo
    overseg = 0
    for k, v in gt_to_algo.items():
        if len(v) > 1:
            overseg += 1

    # Undersegmentation: when algorithm fuses 2 neurons
    underseg = 0
    for k, v in algo_to_gt.items():
        if len(v) > 1:
            underseg += 1

    # TODO save results as variables. Pickle them and save in results folder
    accuracy_results = {'fn': fn_count,
                        'fn_p': fn_p,
                        'fp': fp_count,
                        'fp_p': fp_p,
                        'tp': tp_count,
                        'tp_p': tp_p,
                        'os': overseg,
                        'us': underseg,
                        'algorithm_neurons': np.count_nonzero(np.unique(algo_3d)),
                        'gt_neurons': np.count_nonzero(np.unique(gt_3d)),
                        'gt_to_algo_dict': gt_to_algo,
                        'algo_to_gt_dict': algo_to_gt,
                        'vol_gt': gt_to_algo_volumes,
                        'vol_algo': algo_to_gt_volumes,
                        }

    if verbose >= 1:
        print(f'False negatives: {fn_count} of {len(gt_to_algo.keys())} or {fn_p}%')
        print(f'False positives: {fp_count} of {len(gt_to_algo.keys())} or {fp_p}%')
        print(f'True positives: {tp_count} of {len(gt_to_algo.keys())} or {tp_p}%')
        print(f'There are {overseg} instances of oversegmentation')
        print(f'There are {underseg} instances of undersegmentation')
        print(f'*** Total number of neurons: gt = {len(gt_to_algo)}, algo = {np.count_nonzero(np.unique(algo_3d))} ***')

    return accuracy_results


def create_3d_match_dict(dataset1, dataset2, verbose=0):
    """

    Parameters
    ----------
    dataset1:   numpy array
        3d-numpy array (although 2d should work too).
    dataset2:   numpy array
        3d-numpy array (although 2d should work too).
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    matches_dict: dict
        Dictionary with: keys = neuron ID of dataset1, values = matched neuron IDs of dataset2.
    matches_area: dict
        Dictionary with: keys = neuron ID of dataset1, values = areas of matched neurons of dataset2.

    """
    if verbose >= 1:
        print(f'ID matching start')

    # list of neuron IDs of dataset 1. remove '0'
    dataset1_uniq = np.unique(dataset1).astype(int)
    dataset1_uniq = dataset1_uniq[dataset1_uniq > 0]

    interim_match = list()
    interim_volume = list()
    matches_dict = dict.fromkeys(dataset1_uniq)
    volumes_dict = dict.fromkeys(dataset1_uniq)

    for neuron in dataset1_uniq:
        # loop over GT slices and check for neuron
        if neuron == 0:
            continue
        if int(neuron) % 50 == 0 and verbose >= 1:
            print(f'Neuron: {neuron}')

        # create 3D mask of neuron
        this_mask = dataset1 == neuron

        # check same slice of dataset2 and save all unique IDs, if 'this_mask' has matched anything
        # match_mask = this_mask * dataset2
        match_mask = dataset2[this_mask]

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
            interim_match.append(id.astype(int))  # add IDs
            interim_volume.append(this_overlap_sum)  # add matched area size

        # add ids and areas to dicts
        matches_dict[neuron] = interim_match
        volumes_dict[neuron] = interim_volume

        # clear interim
        interim_match = list()
        interim_volume = list()

    return matches_dict, volumes_dict


def remove_small_overlaps(matches, matches_volumes, array, threshold=20, verbose=0):
    """
    Removes entries in input-dictionaries, if the overlapping volume of 2 neuron masks (volume is in 'matches_volumes')
    is greater than the threshold; threshold = % of volume of neuron to, which matches were matched.
    I.e.: if 5 neurons of dataset 2 were matched to "neuron 1" of dataset 1, then it will remove matches, if
    the matched volumes are smaller than 20% of the volume of 'neuron 1' in dataset 1

    Parameters
    ----------
    matches : dict
        {neuron# : list(IDs of matching neurons)}
    matches_volumes : dict
        {neuron# : list(volumes of matching neurons)}
    array : 3d array
        array of 'dataset 1', i.e. array to which a second dataset has been matched
    threshold : int
        Cutoff for matched volumes; they need to be bigger than 'threshold' % of volume of 'neuron #' in array
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    matches : dict
    matches_volumes : dict
    In both dictionaries, entries were removed, if overlapping volume < threshold

    """
    if verbose >= 1:
        print(f'Removing overlaps, which cover <{threshold}%')
    counter = 0
    for k, v in matches_volumes.items():
        if len(v) > 1:
            vol_k = np.count_nonzero(array == int(k))
            to_keep = [i for i, x in enumerate(v) if ((x / vol_k) * 100) > threshold]
            new_vals = [x for x in v if ((x / vol_k) * 100) > threshold]
            matches_volumes[k] = new_vals
            matches[k] = [m for i, m in enumerate(matches[k]) if i in to_keep]

    return matches, matches_volumes
