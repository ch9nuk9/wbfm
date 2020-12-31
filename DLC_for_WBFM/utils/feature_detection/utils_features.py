# Use classical methods for building and matching features
import os
import numpy as np
from DLC_for_WBFM.utils.point_clouds.utils_bcpd_segmentation import bcpd_to_pixels, pixels_to_bcpd
import cv2
import open3d as o3d
from scipy import stats
import tifffile
import copy

##
## First, extract features and match
##

def convert_to_grayscale(im1):
    try:
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    except:
        im1Gray = im1
    return im1Gray


def detect_features(im1, im2, max_features):

    im1Gray = convert_to_grayscale(im1)
    im2Gray = convert_to_grayscale(im2)

    orb = cv2.ORB_create(max_features)
    kp1, d1 = orb.detectAndCompute(im1Gray, None)
    kp2, d2 = orb.detectAndCompute(im2Gray, None)

    return kp1, d1, kp2, d2


def detect_features_and_match(im1, im2,
                              max_features=3000,
                              matches_to_keep=0.2):
    """
    Uses orb to detect and match generic features
    """

    keypoints1, descriptors1, keypoints2, descriptors2 = detect_features(im1, im2, max_features)
    if len(keypoints1)==0 or len(keypoints2)==0:
        print("Found no keypoints on at least one frame; skipping")
        return keypoints1, keypoints2, []

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * matches_to_keep)
    matches = matches[:numGoodMatches]

    return keypoints1, keypoints2, matches


def match_using_known_keypoints(im1, kp1, im2, kp2, max_features=1000, use_flann=False):
    """ Match using previously detected points, e.g. neurons

    Parameters
    ----------
    im1 : array
        First image
    kp1 : list of cv2.KeyPoint
        First image's keypoints
    im2 : array
        Second image
    kp2 : list of cv2.KeyPoint
        Keypoints
    max_features : int
        number of features; NOT USED
    use_flann : bool
        option for fancier optimizer... BROKEN

    Returns
    -------
    kp1, kp2, matches
        keypoints and matches

    """

    # Match features.
    if use_flann:
        # TODO: not working!
        # Initiate SIFT detector
        sift = cv2.SIFT()

        # use the keypoints to find descriptors with SIFT
        _, descriptors1 = sift.compute(im1,kp1)
        _, descriptors2 = sift.compute(im2,kp2)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(descriptors1,descriptors2,k=1)

    else:
        orb = cv2.ORB_create(max_features)
        _, descriptors1 = orb.compute(im1, kp1)
        _, descriptors2 = orb.compute(im2, kp2)
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    #     matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
        matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
#     numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
#     matches = matches[:numGoodMatches]

    return kp1, kp2, matches


##
## Second, combine with neuron segmentations (one plane)
##

def extract_location_of_matches(matches, keypoints1, keypoints2):
    """Gets location from cv2 objects

    Parameters
    ----------
    matches : list of cv2.match
        Determined e.g. using orb
    keypoints1 : list of cv2.Keypoint
    keypoints2 : list of cv2.Keypoint

    Returns
    -------
    points1, points2

    """

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return points1, points2


def build_neuron_tree(neurons, to_mirror=True):
    """
    Build neuron point cloud from a list of 3d neuron positions
        Expected input syntax: ZXY
    """
    pc = o3d.geometry.PointCloud()

    num_neurons = neurons.shape[0]
    # the segmentations are mirrored
    if to_mirror:
        flip = lambda n : np.array([n[0], n[2], n[1]])
        neurons = np.array([flip(row) for row in neurons])

    # Build point cloud and tree
    pc.points = o3d.utility.Vector3dVector(neurons)
    pc_tree = o3d.geometry.KDTreeFlann(pc)

    return num_neurons, pc, pc_tree


def build_feature_tree(features, which_slice=None):
    """
    Build feature point cloud
        If the input is 2d, then a 3rd dimension is added to form:
        ZXY
    """
    pc = o3d.geometry.PointCloud()

    num_features = features.shape[0]

    # Add 3rd dimension to the features
    if len(features[0])==2:
        if which_slice is None:
            print("Must pass z slice information if features are 2d")
            raise ValueError
        features_3d = np.array([np.hstack((which_slice, row)) for row in features])
    else:
        features_3d = features

    # Build point cloud and tree
    pc.points = o3d.utility.Vector3dVector(features_3d)
    pc_tree = o3d.geometry.KDTreeFlann(pc)

    return num_features, pc, pc_tree


def keep_best_match(all_matches, all_confidences, verbose=0):
    # Get duplicates
    match_array = np.array(all_matches)
    vals, counts = np.unique(match_array[:,1], return_counts=True)
    duplicate_vals = vals[counts>1]

    to_remove = []
    for val in duplicate_vals:
        # Find duplicate matches
        these_duplicates = np.where(match_array[:,1]==val)[0]
        # Get highest confidence value, and keep it
        best_match = np.argmax(np.array(all_confidences)[these_duplicates])
        if verbose >= 2:
            print(f"Keeping best match {best_match} among confidences {np.array(all_conf)[these_duplicates]}")
        these_duplicates = np.delete(these_duplicates,best_match)

        to_remove.extend(these_duplicates)

    to_remove.sort(reverse=True)
    if verbose >= 1:
        print(f"Removed the following duplicates: {to_remove}")
    for i in to_remove:
        all_matches.pop(i)
        all_confidences.pop(i)

    return all_matches, all_confidences

def match_centroids_using_tree(neurons0,
                               neurons1,
                               features0,
                               features1,
                               radius=1,
                               max_nn=10,
                               min_features_needed=1,
                               verbose=0,
                               to_mirror=True,
                               which_slice=None,
                               only_keep_best_match=True):
    """
    Uses a combined point cloud (neurons and features) to do the following:
    1. Assign features, f0, in vol0 to neurons in vol0, n0
    2. Match features f0 to features in vol1, f1
    3. Find the single neuron in vol1, n1, matching most of the features, f1
    """

    # Build point clouds and trees
    num_features0, pc_f0, tree_features0 = build_feature_tree(features0,which_slice)
    num_features1, pc_f1, tree_features1 = build_feature_tree(features1,which_slice)
    num_neurons0, pc_n0, _ = build_neuron_tree(neurons0, to_mirror)
    num_neurons1, pc_n1, tree_neurons1 = build_neuron_tree(neurons1, to_mirror)

    # First, build dictionary to translate features to neurons
    features_to_neurons1 = np.zeros(len(features1))
    for i in range(num_features1):
        # Get features of this neuron and save
        this_feature = np.asarray(pc_f1.points)[i]
        [k, this_fn1, _] = tree_neurons1.search_hybrid_vector_3d(this_feature, radius=5*radius, max_nn=1)

        if k>0:
            features_to_neurons1[i] = this_fn1[0]

        if verbose >= 4:
            # Note: displays the full image
            pc_f0.paint_uniform_color([0.5, 0.5, 0.5])

            one_point = o3d.geometry.PointCloud()
            one_point.points = o3d.utility.Vector3dVector([this_feature])
            one_point.paint_uniform_color([1,0,0])

            np.asarray(pc_f0.colors)[this_f0[1:], :] = [0, 1, 0]
            o3d.visualization.draw_geometries([one_point,pc_f0])

    # Second, loop through neurons of first frame
    all_matches = []
    confidence_func = lambda matches, total : matches / (9+total)
    all_confidences = []
    for i in range(neurons0.shape[0]):
        # Get features of this neuron
        this_neuron = np.asarray(pc_n0.points)[i]
        [_, this_f0, _] = tree_features0.search_hybrid_vector_3d(this_neuron, radius=radius, max_nn=max_nn)

        if verbose >= 3:
            pc_f0.paint_uniform_color([0.5, 0.5, 0.5])

            one_point = o3d.geometry.PointCloud()
            one_point.points = o3d.utility.Vector3dVector([this_neuron])
            one_point.paint_uniform_color([1,0,0])

            np.asarray(pc_f0.colors)[this_f0[1:], :] = [0, 1, 0]

#             print("Visualize the point cloud.")
            o3d.visualization.draw_geometries([one_point,pc_f0])

        # Get the corresponding neurons in vol1, and vote
        this_n1 = features_to_neurons1[this_f0]
        if len(this_n1) >= min_features_needed:
            this_match = int(stats.mode(this_n1)[0][0])
            all_matches.append([i, this_match])
            # Also calculate a heuristic confidence
            num_matches = np.count_nonzero(abs(this_n1-this_match) < 0.1)
            conf = confidence_func(num_matches, len(this_n1))
            all_confidences.append(conf)
            if verbose >= 1:
                print(f"Matched neuron {i} based on {len(this_f0)} features")
        else:
            #all_matches.append([i, np.nan]) # TODO
            #all_confidences.append(0)
            if verbose >= 1:
                print(f"Could not match neuron {i}")

    if only_keep_best_match:
        all_matches, all_confidences = keep_best_match(all_matches, all_confidences, verbose=verbose)

    return all_matches, features_to_neurons1, all_confidences


##
## Extend to full volume
##

# Get images with segmentation


def build_features_on_all_planes(dat0, dat1,
                                verbose=1, start_plane=10,
                                detect_keypoints=True,
                                kp0=None,
                                kp1=None,
                                sz=31.0,
                                num_features_per_plane=1000,
                                matches_to_keep=0.5,
                                dat_foldername = r'..\point_cloud_alignment'):
    """
    Multi-plane wrapper around: match_centroids_using_tree
    """

    all_features0 = []
    all_features1 = []
    for i in range(dat0.shape[0]):
        if i<start_plane:
            continue
        im0 = np.squeeze(dat0[i,...])
        im1 = np.squeeze(dat1[i,...])
        if detect_keypoints:
            #keypoints0, _, keypoints1, _ = detect_features(im1, im2, num_features_per_plane)
            keypoints0, keypoints1, matches = detect_features_and_match(im0, im1, num_features_per_plane, matches_to_keep)
            if len(matches)==0:
                continue
        else:
            kp0_cv2 = get_keypoints_from_3dseg(kp0, i, sz=sz)
            kp1_cv2 = get_keypoints_from_3dseg(kp1, i, sz=sz)
            keypoints0, keypoints1, matches = match_using_known_keypoints(im0, kp0_cv2, im1, kp1_cv2, 1000)
        features0, features1 = extract_location_of_matches(matches, keypoints0, keypoints1)
        # TODO: These features are sorted by matches... is this necessary??

        if verbose >= 3:
            imMatches = cv2.drawMatches(im0, keypoints0, im1, keypoints1, matches, None)
            fig, axis = plt.subplots(figsize=(45, 15))
            plt.imshow(imMatches)

        if verbose >= 2:
            print(f"Adding {len(features0)} features from plane {i}")
#             print(f"Adding {len(features1)} features from plane {i} in volume 1")
        features_3d = np.array([np.hstack((i, row)) for row in features0])
        all_features0.extend(features_3d)
        features_3d = np.array([np.hstack((i, row)) for row in features1])
        all_features1.extend(features_3d)

    return np.array(all_features0), np.array(all_features1), keypoints0, keypoints1


def get_keypoints_from_3dseg(kp0, i, sz=31.0, neuron_height=3):
    """Translate numpy array to cv2.Keypoints, based off one slice

    Parameters
    ----------
    kp0 : array-like
        Original positions
    i : int
        current slice
    sz : float
        Size for cv2 keypoints... not sure if this matters
    neuron_height : float
        Radius around original annotations to add the keypoint to

    Returns
    -------
    type
        Description of returned object.

    """
    kp_cv2 = []
    for z,x,y in kp0:
        if abs(z-i) < neuron_height:
            kp_cv2.append(cv2.KeyPoint(y,x,sz))

    return kp_cv2
