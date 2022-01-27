import numpy as np

from DLC_for_WBFM.utils.neuron_matching.utils_detection import build_point_clouds_for_volume, \
    build_correspondence_icp, get_centroids_from_df
from DLC_for_WBFM.utils.neuron_matching.utils_tracklets import build_tracklets_from_matches


def detect_neurons_using_ICP(dat,
                             num_slices,
                             alpha=1.0,
                             min_detections=3,
                             start_slice=2,
                             verbose=0):
    """
    Use blob detection and ICP to find neurons on multiple planes and link
    """

    # Build point clouds for each plane
    # ENHANCE: Remove alpha from this function
    all_keypoints_pcs = build_point_clouds_for_volume(dat,
                                                  num_slices,
                                                  alpha,
                                                  start_slice=start_slice,
                                                  verbose=verbose)
    if verbose >= 1:
        print("Building pairwise correspondence...")
    all_icp = build_correspondence_icp(all_keypoints_pcs,
                                       verbose=verbose-1)
    if verbose >= 1:
        print("Building clusters...")
    all_neurons = [np.asarray(k.points) for k in all_keypoints_pcs]
    all_matches = [np.asarray(ic.correspondence_set) for ic in all_icp]
    clust_df = build_tracklets_from_matches(all_neurons,
                                            all_matches,
                                            verbose=verbose-1)

    centroids = get_centroids_from_df(clust_df, min_detections, verbose=verbose-1)
    if verbose >= 1:
        print("Finished ID'ing neurons")

    return centroids, clust_df, all_icp, all_keypoints_pcs