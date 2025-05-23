# Use classical methods for building and matching features

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from tqdm.auto import tqdm

from wbfm.utils.external.utils_cv2 import get_keypoints_from_3dseg


##
## First, extract features and match
##
from wbfm.utils.external.custom_errors import NoMatchesError


def convert_to_grayscale(im1):
    try:
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    except:
        im1Gray = im1
    return im1Gray


def detect_keypoints_and_features(im1, max_features,
                                  setFastThreshold=True,
                                  use_sift=False,
                                  detector=None):
    # Assumes 2d image
    im1Gray = convert_to_grayscale(im1)
    if detector is None:
        detector = build_cv2_detector(max_features, setFastThreshold, use_sift)
    kp1, d1 = detector.detectAndCompute(im1Gray, None)

    return kp1, d1


def detect_only_keypoints(im1, max_features,
                          setFastThreshold=True,
                          use_sift=False):
    # Assumes 2d image
    im1Gray = convert_to_grayscale(im1)
    detector = build_cv2_detector(max_features, setFastThreshold, use_sift)
    kp1 = detector.detect(im1Gray, None)

    return kp1


def build_cv2_detector(max_features, setFastThreshold, use_sift):
    if use_sift:
        options = {'hessianThreshold': 0.1}
        detector = cv2.xfeatures2d.SURF_create(**options)
    else:
        detector = cv2.ORB_create(max_features)  # , WTA_K=3)
        if setFastThreshold:
            detector.setFastThreshold(0)
    return detector


def match_known_features(descriptors1, descriptors2,
                         keypoints1=None,
                         keypoints2=None,
                         im1_shape=None,
                         im2_shape=None,
                         matches_to_keep=1.0,
                         use_GMS=True,
                         use_orb=False,
                         crossCheck=True):
    # MAIN USE FUNCTION

    # Match features.
    if use_orb:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        matches = matcher.match(descriptors1, descriptors2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
        matches = matcher.match(descriptors1, descriptors2)

    if use_GMS:
        options = {'keypoints1': keypoints1,
                   'keypoints2': keypoints2,
                   'matches1to2': matches,
                   'withRotation': False,
                   'thresholdFactor': 6.0}
        matches = cv2.xfeatures2d.matchGMS(im1_shape, im2_shape, **options)

    if matches_to_keep < 1.0:
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * matches_to_keep)
        matches = matches[:numGoodMatches]

    return matches


def detect_features_and_match(im1, im2,
                              max_features=3000,
                              matches_to_keep=0.3,
                              use_GMS=True,
                              verbose=0):
    """
    Uses orb to detect and match generic features
    """

    keypoints1, descriptors1 = detect_keypoints_and_features(im1, max_features)
    keypoints2, descriptors2 = detect_keypoints_and_features(im2, max_features)
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        if verbose >= 1:
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
    """ OLD

    Match using previously detected points, e.g. neurons

    # OPTIMIZE: combine with match_known_keypoints()

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
        #  not working!
        # Initiate SIFT detector
        sift = cv2.SIFT()

        # use the keypoints to find descriptors with SIFT
        _, descriptors1 = sift.compute(im1, kp1)
        _, descriptors2 = sift.compute(im2, kp2)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=1)

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

def extract_map1to2_from_matches(matches):
    """
    Get dict from list of cv2 Match objects
    """
    matches_dict = {}
    for i, m in enumerate(matches):
        # Maybe, queryIdx: https://stackoverflow.com/questions/22082598/how-to-get-matches-drawn-by-drawmatches-in-an-array-for-example-in-a-dmatch-s/24183734
        # BUG: may overwrite if not 1-to-1
        matches_dict[m.queryIdx] = m.trainIdx

    return matches_dict


def keep_top_matches_per_neuron(keypoint_matches, frame, matches_to_keep=0.5):
    """
    Postprocesses matches to remove a percentage locally, not globally

    If any neurons start with a match, they will retain at least one
    """

    sz = len(frame.neuron_locs)
    to_keep = []
    # OPTIMIZE: this requires sz loops over all keypoints
    for neuron in range(sz):
        these_keypoints = set(frame.get_features_of_neuron(neuron))
        global_ind_and_dist = [(i, kp.distance) for i, kp in enumerate(keypoint_matches) if
                               kp.queryIdx in these_keypoints]
        if len(global_ind_and_dist) == 0:
            continue
        local_sort_idx = np.argsort(np.array(global_ind_and_dist)[:, 1])
        num_to_keep = max(int(len(local_sort_idx) * matches_to_keep), 1)
        good_local_idx = local_sort_idx[:num_to_keep]
        good_global_idx = np.array(global_ind_and_dist)[good_local_idx, 0].astype(int)
        to_keep.extend(list(good_global_idx))

    return [keypoint_matches[i] for i in to_keep]


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
    import open3d as o3d
    pc = o3d.geometry.PointCloud()

    num_neurons = neurons.shape[0]
    if len(neurons.shape) == 1:
        neurons = np.expand_dims(neurons, 0)
    # the segmentations are mirrored
    if to_mirror:
        flip = lambda n: np.array([n[0], n[2], n[1]])
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
    import open3d as o3d
    pc = o3d.geometry.PointCloud()

    num_features = features.shape[0]

    # Add 3rd dimension to the features
    if len(features[0]) == 2:
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
    vals, counts = np.unique(match_array[:, 1], return_counts=True)
    duplicate_vals = vals[counts > 1]

    to_remove = []
    for val in duplicate_vals:
        # Find duplicate matches
        these_duplicates = np.where(match_array[:, 1] == val)[0]
        # Get highest confidence value, and keep it
        best_match = np.argmax(np.array(all_confidences)[these_duplicates])
        if verbose >= 2:
            print(f"Keeping best match {best_match}")
        these_duplicates = np.delete(these_duplicates, best_match)

        to_remove.extend(these_duplicates)

    to_remove.sort(reverse=True)
    if verbose >= 1:
        print(f"Removed the following duplicates: {to_remove}")
    for i in to_remove:
        all_matches.pop(i)
        all_confidences.pop(i)

    return all_matches, all_confidences, to_remove


def build_f2n_map(features1,
                  num_features1,
                  pc_f1,
                  radius,
                  tree_n1,
                  verbose=0):
    import open3d as o3d
    features_to_neurons1 = dict()
    nn_opt = {'radius': 5 * radius, 'max_nn': 1}
    for i in range(num_features1):
        # Get features of this neuron and save
        this_feature = np.asarray(pc_f1.points)[i]
        [k, this_fn1, _] = tree_n1.search_hybrid_vector_3d(this_feature, **nn_opt)

        if k > 0:
            features_to_neurons1[i] = this_fn1[0]

            if verbose >= 4:
                # Note: displays the full image
                # Default color = gray
                pc_f1.paint_uniform_color([0.9, 0.9, 0.9])
                # Current feature being queried
                one_point = o3d.geometry.PointCloud()
                one_point.points = o3d.utility.Vector3dVector([this_feature])
                # one_point.points = o3d.utility.Vector3dVector(this_neighbor)
                one_point.paint_uniform_color([0, 0, 1])
                # Found closest neuron
                np.asarray(pc_f1.colors)[this_fn1[1:], :] = [0, 1, 0]

                o3d.visualization.draw_geometries([one_point, pc_f1])

    return features_to_neurons1


def add_neuron_match(all_best_matches,
                     all_confidences,
                     i,
                     min_features_needed,
                     this_n1,
                     verbose=0,
                     all_candidate_matches=None):
    """
    Processes an array of neuron matches into a neuron match

    Parameters
    =============
    all_best_matches : list
        List to extend
    all_confidences : list
        List to extend
    i : int
        The index of this neuron (local to this frame)
    min_features_needed : int
        Under this, no match is counted
    this_n1 : list
        List of candidate matches

    if all_candidate_matches is passed, then all candidates are saved
    """
    this_n1 = np.array(this_n1)

    n = len(this_n1)
    get_num_matches = lambda this_match: np.count_nonzero(abs(this_n1 - this_match) < 0.1)
    get_conf = lambda num_matches: num_matches / (1 + n)
    if n >= min_features_needed:
        # Simple plurality voting
        best_match = int(stats.mode(this_n1)[0][0])
        all_best_matches.append([i, best_match])
        # Also calculate a heuristic confidence
        conf = get_conf(get_num_matches(best_match))
        all_confidences.append(conf)
        if all_candidate_matches is not None:
            vals, counts = np.unique(this_n1, return_counts=True)
            new_candidates = [(i, v, get_conf(c)) for v, c in zip(vals, counts)]
            all_candidate_matches.extend(new_candidates)
        # if verbose >= 1:
        #     print(f"Matched neuron {i} to {this_match} based on {len(this_n1)} matches")
    else:
        # all_matches.append([i, np.nan])
        # all_confidences.append(0)
        if verbose >= 1:
            print(f"Could not match neuron {i}")

    return all_best_matches, all_confidences, all_candidate_matches


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

    nn_opt = {'radius': radius, 'max_nn': max_nn}

    all_matches = []
    all_confidences = []
    for i in range(neurons0.shape[0]):
        # Get features of this neuron
        if pc_n0 is None:
            this_neuron = neurons0[i, :]
        else:
            # I use pc_n0 because there may be a coordinate transformation in the pc
            this_neuron = np.asarray(pc_n0.points)[i]
        [_, this_f0, _] = tree_features0.search_hybrid_vector_3d(this_neuron, **nn_opt)

        # if verbose >= 3:
        #     pc_f0.paint_uniform_color([0.5, 0.5, 0.5])
        #
        #     one_point = o3d.geometry.PointCloud()
        #     one_point.points = o3d.utility.Vector3dVector([this_neuron])
        #     one_point.paint_uniform_color([1, 0, 0])
        #
        #     np.asarray(pc_f0.colors)[this_f0[1:], :] = [0, 1, 0]
        #
        #     #             print("Visualize the point cloud.")
        #     o3d.visualization.draw_geometries([one_point, pc_f0])

        # Get the corresponding neurons in vol1, and vote
        f2n = features_to_neurons1
        this_n1 = [f2n[f1] for f1 in this_f0 if f1 in f2n]

        all_matches, all_confidences, _ = add_neuron_match(
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
    num_features0, pc_f0, tree_features0 = build_feature_tree(features0, which_slice)
    num_features1, pc_f1, tree_features1 = build_feature_tree(features1, which_slice)
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
        all_matches, all_confidences, _ = keep_best_match(all_matches, all_confidences, verbose=verbose)

    return all_matches, features_to_neurons1, all_confidences


##
## Extend to full volume
##

# Get images with segmentation

def build_features_and_match_2volumes(dat0, dat1,
                                      verbose=1, start_plane=10,
                                      detect_new_keypoints=True,
                                      kp0_zxy=None,
                                      kp1_zxy=None,
                                      sz=31.0,
                                      num_features_per_plane=1000,
                                      matches_to_keep=0.5,
                                      use_GMS=True,
                                      dat_foldername=r'..\point_cloud_alignment'):
    """
    Multi-plane wrapper around: detect_features_and_match

    Returns a list of lists, with the outer list referring to the plane

    Note: the matches refer to the keypoint list, not the locations (which are already matched)
    """

    all_locs0 = []
    all_locs1 = []
    all_matches = []
    all_kp0 = []
    all_kp1 = []
    all_match_offsets = [[0, 0]]
    if start_plane > dat0.shape[0]:
        print("Warning: Start plane is greater than the shape of the image... no matches possible")
    for i in tqdm(range(dat0.shape[0]), leave=False):
        if i < start_plane:
            continue
        im0 = np.squeeze(dat0[i, ...])
        im1 = np.squeeze(dat1[i, ...])
        if detect_new_keypoints:
            # keypoints0, _, keypoints1, _ = detect_features(im1, im2, num_features_per_plane)
            keypoints0, keypoints1, matches = detect_features_and_match(im0, im1, num_features_per_plane,
                                                                        matches_to_keep, use_GMS)
            if len(matches) == 0:
                continue
        else:
            assert kp0_zxy is not None, 'Must pass old keypoints if detect_new_keypoints=False'
            kp0_cv2 = get_keypoints_from_3dseg(kp0_zxy, i, sz=sz)
            kp1_cv2 = get_keypoints_from_3dseg(kp1_zxy, i, sz=sz)
            keypoints0, keypoints1, matches = match_using_known_keypoints(im0, kp0_cv2, im1, kp1_cv2, 1000)
        locs0, locs1 = extract_location_of_matches(matches, keypoints0, keypoints1)

        if verbose >= 3:
            imMatches = cv2.drawMatches(im0, keypoints0, im1, keypoints1, matches, None)
            fig, axis = plt.subplots(figsize=(45, 15))
            plt.imshow(imMatches)

        if verbose >= 2:
            print(f"Adding {len(locs0)} locations from plane {i}")
        locs_3d_0 = np.array([np.hstack((i, row)) for row in locs0])
        all_locs0.extend(locs_3d_0)
        locs_3d_1 = np.array([np.hstack((i, row)) for row in locs1])
        all_locs1.extend(locs_3d_1)
        all_matches.append(matches)
        all_kp0.extend(keypoints0)
        all_kp1.extend(keypoints1)
        all_match_offsets.append([len(all_locs0), len(all_locs1)])

    if len(all_matches) == 0:
        raise NoMatchesError("No matches found on any planes; "
                             "probably, the z depth was set incorrectly or these images are empty")

    return np.array(all_locs0), np.array(all_locs1), all_kp0, all_kp1, all_matches, all_match_offsets
