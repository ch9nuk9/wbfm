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


def detect_features(im1, max_features):

    im1Gray = convert_to_grayscale(im1)
    orb = cv2.ORB_create(max_features)
    kp1, d1 = orb.detectAndCompute(im1Gray, None)

    return kp1, d1


def match_known_features(descriptors1, descriptors2,
                         keypoints1=None,
                         keypoints2=None,
                         im1_shape=None,
                         im2_shape=None,
                         matches_to_keep=1.0,
                         use_GMS=True):

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)

    if use_GMS:
        opt = {'keypoints1':keypoints1, 'keypoints2':keypoints2, 'matches1to2':matches}
        matches = cv2.xfeatures2d.matchGMS(im1_shape, im2_shape, **opt)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * matches_to_keep)
    matches = matches[:numGoodMatches]

    return matches


def detect_features_and_match(im1, im2,
                              max_features=3000,
                              matches_to_keep=0.2,
                              use_GMS=True):
    """
    Uses orb to detect and match generic features
    """

    keypoints1, descriptors1 = detect_features(im1, max_features)
    keypoints2, descriptors2 = detect_features(im2, max_features)
    if len(keypoints1)==0 or len(keypoints2)==0:
        print("Found no keypoints on at least one frame; skipping")
        return keypoints1, keypoints2, []

    matches = match_known_features(descriptors1, descriptors2,
                                   keypoints1,
                                   keypoints2,
                                   im1.shape,
                                   im2.shape,
                                   matches_to_keep=matches_to_keep,
                                   use_GMS=use_GMS)

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

def extract_indices_of_matches(matches):
    """
    Get np.array from list of cv2 Match objects
    """
    matches_array = np.zeros((len(matches),2),dtype=int)

    for i, m in enumerate(matches):
        matches_array[i,0] = m.trainIdx
        matches_array[i,1] = m.queryIdx

    return matches_array

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


def build_f2n_map(features1,
                   num_features1,
                   pc_f1,
                   radius,
                   tree_n1,
                   verbose=0):
    features_to_neurons1 = np.zeros(len(features1))
    nn_opt = { 'radius':5*radius, 'max_nn':1}
    for i in range(num_features1):
        # Get features of this neuron and save
        this_feature = np.asarray(pc_f1.points)[i]
        [k, this_fn1, _] = tree_n1.search_hybrid_vector_3d(this_feature, **nn_opt)

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

    return features_to_neurons1


def add_neuron_match(all_neuron_matches,
                    all_confidences,
                    i,
                    min_features_needed,
                    this_n1,
                    verbose):

    confidence_func = lambda matches, total : matches / (9+total)
    if len(this_n1) >= min_features_needed:
        this_match = int(stats.mode(this_n1)[0][0])
        all_neuron_matches.append([i, this_match])
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

    return all_neuron_matches, all_confidences

def calc_2frame_matches(neurons0,
                        tree_features0,
                        features_to_neurons1,
                        radius,
                        max_nn,
                        min_features_needed,
                        pc_n0=None,
                        verbose=0):
    """
    Calculates the matches between the features of two frames and translates to
    neuron space
    """

    nn_opt = {'radius':radius, 'max_nn':max_nn}

    all_matches = []
    all_confidences = []
    for i in range(neurons0.shape[0]):
        # Get features of this neuron
        if pc_n0 is None:
            this_neuron = neurons0[i,:]
        else:
            # I use pc_n0 because there may be a coordinate transformation in the pc
            this_neuron = np.asarray(pc_n0.points)[i]
        [_, this_f0, _] = tree_features0.search_hybrid_vector_3d(this_neuron, **nn_opt)

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

        all_matches, all_confidences = add_neuron_match(
            all_matches,
            all_confidences,
            i,
            min_features_needed,
            this_n1,
            verbose
        )

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

    Note: Assumes the features are already matched by ordering
    """

    # Build point clouds and trees
    num_features0, pc_f0, tree_features0 = build_feature_tree(features0,which_slice)
    num_features1, pc_f1, tree_features1 = build_feature_tree(features1,which_slice)
    num_neurons0, pc_n0, _ = build_neuron_tree(neurons0, to_mirror)
    num_neurons1, pc_n1, tree_neurons1 = build_neuron_tree(neurons1, to_mirror)

    # First, build array to translate features to neurons
    features_to_neurons1 = build_f2n_map(features1,
                                           num_features1,
                                           pc_f1,
                                           radius,
                                           tree_neurons1,
                                           verbose=0)

    # Second, loop through neurons of first frame and match
    all_matches, all_confidences = calc_2frame_matches(neurons0,
                                                        tree_features0,
                                                        features_to_neurons1,
                                                        radius,
                                                        max_nn,
                                                        min_features_needed,
                                                        pc_n0=pc_n0,
                                                        verbose=0)

    if only_keep_best_match:
        all_matches, all_confidences = keep_best_match(all_matches, all_confidences, verbose=verbose)

    return all_matches, features_to_neurons1, all_confidences


##
## Extend to full volume
##

# Get images with segmentation

def build_features_1volume(dat, num_features_per_plane=1000, verbose=0):

    all_features = []
    all_locs = []
    all_kps = []
    for i in range(dat.shape[0]):
        im = np.squeeze(dat[i,...])
        kp, features = detect_features(im, num_features_per_plane)

        if features is None:
            continue
        all_features.extend(features)
        all_kps.extend(kp)
        locs_3d = np.array([np.hstack((i, row.pt)) for row in kp])
        all_locs.extend(locs_3d)

    return all_kps, np.array(all_locs), np.array(all_features)



def build_features_and_match_2volumes(dat0, dat1,
                                verbose=1, start_plane=10,
                                detect_keypoints=True,
                                kp0=None,
                                kp1=None,
                                sz=31.0,
                                num_features_per_plane=1000,
                                matches_to_keep=0.5,
                                dat_foldername = r'..\point_cloud_alignment'):
    """
    Multi-plane wrapper around: detect_features_and_match
    """

    all_locs0 = []
    all_locs1 = []
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
        locs0, locs1 = extract_location_of_matches(matches, keypoints0, keypoints1)

        if verbose >= 3:
            imMatches = cv2.drawMatches(im0, keypoints0, im1, keypoints1, matches, None)
            fig, axis = plt.subplots(figsize=(45, 15))
            plt.imshow(imMatches)

        if verbose >= 2:
            print(f"Adding {len(locs0)} locations from plane {i}")
        locs_3d = np.array([np.hstack((i, row)) for row in locs0])
        all_locs0.extend(locs_3d)
        locs_3d = np.array([np.hstack((i, row)) for row in locs1])
        all_locs1.extend(locs_3d)

    return np.array(all_locs0), np.array(all_locs1), keypoints0, keypoints1


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
