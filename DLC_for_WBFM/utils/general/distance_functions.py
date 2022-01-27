import numpy as np


def calc_global_track_to_tracklet_distances(this_global_track: np.ndarray, list_tracklets_zxy: list,
                                            min_overlap: int = 0, outlier_threshold=None):
    """For one DLC neuron, calculate distances between that track and all tracklets"""

    all_dist = []
    for this_tracklet in list_tracklets_zxy:
        vec_of_dists, has_overlap = calc_dist_if_overlap(this_tracklet, min_overlap, this_global_track)
        if has_overlap and outlier_threshold is not None:
            vec_of_dists = vec_of_dists > outlier_threshold

        all_dist.append(vec_of_dists)
    return all_dist


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


def calc_confidence_from_distance_array_and_matches(distance_matrix, matches, gamma=1.0):
    # Calculate confidences from distance
    conf = np.zeros((matches.shape[0], 1))
    for i, (m0, m1) in enumerate(matches):
        dist = distance_matrix[m0, m1]
        conf[i] = dist2conf(dist, gamma)
    return conf
