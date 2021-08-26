import pickle

import cv2
import numpy as np
import open3d as o3d

from DLC_for_WBFM.utils.feature_detection.utils_tracklets import build_tracklets_from_matches


def detect_blobs(im1_raw):
    """
    Detects neuron-like blobs in a 2d image
    """
    im1 = cv2.GaussianBlur(im1_raw, (5, 5), 0)
    # im1 = cv2.bilateralFilter(im1_raw, 5, 0, 3)

    im1 = cv2.bitwise_not(im1)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = np.max(im1)
    params.thresholdStep = 1

    params.minDistBetweenBlobs = 2

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 25

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    # detector = cv2.SimpleBlobDetector(params)
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im1)

    return keypoints, im1


def build_point_clouds_for_volume(dat,
                                  num_slices,
                                  alpha,
                                  start_slice=2,
                                  verbose=0):
    """
    Build point clouds for each plane, with points = neurons
    """

    all_keypoints_pcs = []

    f = lambda dat, which_slice: (alpha * dat[which_slice]).astype('uint8')

    for i in range(num_slices):
        if i < start_slice:
            continue
        im1_raw = f(dat, i)
        kp, im1 = detect_blobs(im1_raw)
        # Add to make the format: ZXY
        kp_3d = np.array([np.hstack((i, row.pt)) for row in kp])

        pc = o3d.geometry.PointCloud()
        if len(kp_3d) > 0:
            pc.points = o3d.utility.Vector3dVector(kp_3d)

        all_keypoints_pcs.append(pc)

    return all_keypoints_pcs


def build_correspondence_icp(all_keypoints_pcs,
                             options=None,
                             verbose=0):
    # Build correspondence between each pair of planes

    if options is None:
        options = {'max_correspondence_distance': 4.0}
    all_icp = []

    for i in range(len(all_keypoints_pcs) - 1):
        if verbose >= 1:
            print(f"{i} / {len(all_keypoints_pcs)}")
        this_pc = all_keypoints_pcs[i]
        next_pc = all_keypoints_pcs[i + 1]

        reg = o3d.pipelines.registration.registration_icp(this_pc, next_pc, **options)

        all_icp.append(reg)

    return all_icp


def get_centroids_from_df(clust_df, min_detections=3, verbose=0):
    # Remove clusters that aren't long enough
    f = lambda x: (len(x) > min_detections)
    valid_detections = clust_df['all_ind_local'].apply(f)
    if verbose >= 1:
        num_not_valid = len(np.where(~valid_detections))
        print(f"Removing {num_not_valid} detections of length < {min_detections}")

    f = lambda x: np.mean(x, axis=0)
    centroids = clust_df.loc[valid_detections, 'all_xyz'].apply(f)

    return centroids


##
## Full function
##

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
                                       verbose=verbose - 1)
    if verbose >= 1:
        print("Building clusters...")
    all_neurons = [np.asarray(k.points) for k in all_keypoints_pcs]
    all_matches = [np.asarray(ic.correspondence_set) for ic in all_icp]
    clust_df = build_tracklets_from_matches(all_neurons,
                                            all_matches,
                                            verbose=verbose - 1)

    centroids = get_centroids_from_df(clust_df, min_detections, verbose=verbose - 1)
    if verbose >= 1:
        print("Finished ID'ing neurons")

    return centroids, clust_df, all_icp, all_keypoints_pcs


##
## Alternative: read from pipeline
##

def detect_neurons_from_file(detection_fname: str, which_volume: int, verbose: int = 0) -> list:
    """
    Designed to be used with centroids detected using a different pipeline
    """
    # dat = pd.read_pickle(detection_fname)
    with open(detection_fname, 'rb') as f:
        # Note: dict of dataframes
        neuron_locs = pickle.load(f)[which_volume]['centroids']
    # In current format: flipped x and y
    neuron_locs = np.array([n for n in neuron_locs])
    if len(neuron_locs) > 0:
        neuron_locs = neuron_locs[:, [0, 2, 1]]
    else:
        neuron_locs = []
    return neuron_locs
