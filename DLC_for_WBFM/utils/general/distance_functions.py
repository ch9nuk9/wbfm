import logging

import numpy as np
from tqdm.auto import tqdm


def calc_global_track_to_tracklet_distances(this_global_track: np.ndarray, list_tracklets_zxy: list,
                                            min_overlap: int = 0, outlier_threshold=None):
    """For one globally tracked neuron, calculate distances between that track and all tracklets"""

    all_dist = []
    for this_tracklet in list_tracklets_zxy:
        vec_of_dists, has_overlap = calc_dist_if_overlap(this_tracklet, min_overlap, this_global_track)
        if has_overlap and outlier_threshold is not None:
            vec_of_dists = vec_of_dists > outlier_threshold

        all_dist.append(vec_of_dists)
    return all_dist


def calc_global_track_to_tracklet_distances_subarray(this_global_track: np.ndarray,
                                                     dict_tracklets_zxy_subarray: dict, dict_tracklets_zxy_ind: dict,
                                                     min_overlap: int = 0, outlier_threshold=None):
    """For one globally tracked neuron, calculate distances between that track and all tracklets"""

    all_dist = []
    names = list(dict_tracklets_zxy_subarray.keys())
    for i, key in enumerate(names):
    # for i, (this_tracklet, tracklet_ind) in enumerate(zip(dict_tracklets_zxy_subarray, dict_tracklets_zxy_ind)):
        this_tracklet = dict_tracklets_zxy_subarray[key]
        tracklet_ind = dict_tracklets_zxy_ind[key]
        vec_of_dists, has_overlap = calc_dist_if_overlap(this_tracklet, min_overlap,
                                                         this_global_track[tracklet_ind[0]:tracklet_ind[1], :])
        if has_overlap and outlier_threshold is not None:
            vec_of_dists = vec_of_dists > outlier_threshold

        all_dist.append(vec_of_dists)
    return all_dist


def precalculate_lists_from_dataframe(all_tracklet_names, coords, df_tracklets, min_overlap):
    """
    Calculates the zxy positions of each tracklet, and the non-nan indices

    Outputs two separate dictionaries, indexed by the tracklet name (example: 'tracklet_0000000')
    """
    dict_tracklets_zxy_small = {}
    dict_tracklets_zxy_ind = {}
    for name in tqdm(all_tracklet_names):
        # Note: can't just dropna because there may be gaps in the tracklet
        tmp = df_tracklets[name][coords]
        idx0, idx1 = tmp.first_valid_index(), tmp.last_valid_index()
        if idx0 is not None and idx1 - idx0 > min_overlap:
            dict_tracklets_zxy_ind[name] = [idx0, idx1 + 1]
            dict_tracklets_zxy_small[name] = tmp.to_numpy()[idx0:idx1 + 1, :]
    _name = list(dict_tracklets_zxy_small.keys())[0]
    # logging.info(f"Precalculated tracklet zxy with shape: {dict_tracklets_zxy_small[_name].shape}")
    return dict_tracklets_zxy_small, dict_tracklets_zxy_ind


def calc_dist_if_overlap(this_tracklet: np.ndarray, min_overlap: int, this_global_track: np.ndarray):
    this_diff = this_tracklet - this_global_track

    # Check for enough common data points
    # num_common_pts = this_diff['x'].notnull().sum()
    num_common_pts = np.count_nonzero(~np.isnan(this_diff[:, 0]))
    if num_common_pts >= min_overlap:
        vec_of_dists = np.linalg.norm(this_diff, axis=1)
        has_overlap = True
    else:
        vec_of_dists = np.inf
        has_overlap = False
    return vec_of_dists, has_overlap


def summarize_distances_quantile(all_dists):
    out = []
    for d in all_dists:
        if np.isscalar(d) and not np.isfinite(d):
            out.append(np.nan)
            continue
        out.append(np.nanquantile(d, 0.1))
    return np.array(out)
    # return list(map(lambda x: np.nanquantile(x, 0.1), all_dists))


def summarize_confidences_outlier_percent(all_dists, outlier_threshold=1.0):
    out = []
    for d in all_dists:
        if np.isscalar(d) and not np.isfinite(d):
            out.append(np.nan)
            continue
        d = d[~np.isnan(d)]
        len_d = len(d)
        if len_d == 0:
            conf = np.nan
        else:
            percent_inliers = np.sum(d < outlier_threshold) / len_d
            conf = confidence_using_tracklet_lengths(len_d, percent_inliers)
        out.append(conf)
    return np.array(out)


def dist2conf(dist, gamma=1.0):
    return np.tanh(gamma / (dist + 1e-3))


def confidence_using_tracklet_lengths(length, percent_inliers,
                                      gamma_length=20, gamma_inliers=0.3):
    """Low confidence if there are few frames, even if all of them are inliers"""
    return percent_inliers * np.tanh(length / gamma_length)
    # return np.tanh(percent_inliers / gamma_inliers) * np.tanh(length / gamma_length)


def calc_confidence_from_distance_array_and_matches(distance_matrix, matches, gamma=1.0, use_dist2conf=True):
    # Calculate confidences from distance
    conf = np.zeros((matches.shape[0], 1))
    for i, (m0, m1) in enumerate(matches):
        dist = distance_matrix[m0, m1]
        if use_dist2conf:
            conf[i] = dist2conf(dist, gamma)
        else:
            conf[i] = dist
    return conf
