import open3d as o3d
import cv2
import numpy as np
from typing import Dict

from DLC_for_WBFM.utils.feature_detection.class_reference_frame import ReferenceFrame
from DLC_for_WBFM.utils.feature_detection.utils_features import build_feature_tree, build_neuron_tree
from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_from_distance, calc_icp_matches
from tqdm import tqdm


def propagate_via_affine_model(which_neuron: int,
                               f0: ReferenceFrame,
                               f1: ReferenceFrame,
                               all_feature_matches: Dict,
                               radius=10.0,
                               min_matches=100,
                               no_match_mode='negative_position',
                               verbose=0):
    """
    1. Gets a cloud of features around a neuron (first frame)
    2. Fits an affine model based on the feature matches (second frame)
    3. Propagates the neuron to a final position

    If the affine fitting fails, then a dummy point is added according to 'no_match_mode'
    The default is to add a point add [-10, -10, -10]
    (this keeps the indices of the pushed point cloud aligned with the original)
    """

    # Get a neuron, then get the features around it
    global close_features, pts0, pts1
    this_neuron = f0.neuron_locs[which_neuron]

    num_features, pc0, tree_features0 = build_feature_tree(f0.keypoint_locs)
    pc0.paint_uniform_color([0.9, 0.9, 0.9])

    # See also calc_2frame_matches
    # Iteratively increases the radius if not enough matches are found
    for i in range(5):
        nn_opt = {'radius': radius, 'max_nn': 5000}
        [_, close_features, _] = tree_features0.search_hybrid_vector_3d(np.asarray(this_neuron), **nn_opt)

        # Get the next-frame-matches of the features in this cloud
        # Get just these points, and align two lists
        pts0, pts1 = [], []
        for match in all_feature_matches:
            v0_ind = match.queryIdx
            if v0_ind in close_features:
                pts0.append(f0.keypoint_locs[v0_ind])
                pts1.append(f1.keypoint_locs[match.trainIdx])
        pts0, pts1 = np.array(pts0), np.array(pts1)

        if len(pts0) < min_matches:
            radius = radius * 1.5
        else:
            break

    np.asarray(pc0.colors)[close_features[1:], :] = [0, 1, 0]

    # Now calculate and apply the affine transformation
    if no_match_mode is 'negative_position':
        neuron0_trans = np.array([[-10.0, -10.0, -10.0]])
    else:
        neuron0_trans = None
    success = False
    if (len(pts0) > 2) & (len(pts1) > 2):
        val, h, inliers = cv2.estimateAffine3D(pts0, pts1, confidence=0.999)
        if verbose >= 2:
            print(f"Found {len(pts0)} matches from {len(close_features)} features")

        if h is not None:
            # And translate the neuron itself
            neuron_matrix = f0.neuron_locs[which_neuron:which_neuron + 1]
            neuron0_trans = cv2.transform(np.array([neuron_matrix]), h)[0]

            success = True
    else:
        if verbose >= 2:
            print("Failed to find a match")

    return success, neuron0_trans


def propagate_all_neurons(f0: ReferenceFrame, f1: ReferenceFrame, all_feature_matches,
                          radius=10.0,
                          min_matches=100,
                          verbose=0):
    """
    Loops over neurons in f0 (frame 0), and applies:
        propagate_via_affine_model(which_neuron, f0, f1, all_feature_matches)
    """
    all_propagated = None

    options = {'f0': f0, 'f1': f1, 'all_feature_matches': all_feature_matches,
               'radius': radius, 'min_matches': min_matches}
    for which_neuron in range(len(f0.neuron_locs)):
        success, n0_propagated = propagate_via_affine_model(which_neuron, **options)
        # Note: needs the failed neurons to keep the indices aligned
        # if not success:
        #     continue
        pc1_propagated = o3d.geometry.PointCloud()
        pc1_propagated.points = o3d.utility.Vector3dVector(n0_propagated)

        if all_propagated is None:
            all_propagated = pc1_propagated
        else:
            all_propagated = all_propagated + pc1_propagated

    return all_propagated


def calc_matches_using_affine_propagation(f0, f1, all_feature_matches,
                                          radius=10.0,
                                          min_matches=100,
                                          maximum_distance=15.0,
                                          verbose=0,
                                          DEBUG=False):
    """
    Propagates the neuron cloud in f0 using all_feature_matches
    Matches to neurons in f1 if the neighbors are close enough

    See also:
        calc_matches_using_feature_voting
        calc_matches_using_gaussian_process
    """

    all_propagated = propagate_all_neurons(f0, f1, all_feature_matches,
                                           radius=radius,
                                           min_matches=min_matches)

    # Loop over locations of pushed v0 neurons
    # out = calc_matches_using_2nn(all_propagated, f1.neuron_locs, distance_ratio=distance_ratio)
    # all_matches, all_conf, all_candidate_matches = out

    xyz0 = np.array(all_propagated.points)
    xyz1 = f1.neuron_locs

    # out = calc_bipartite_from_distance(xyz0, xyz1, max_dist=10*distance_ratio)
    # TODO: Better max distance
    out = calc_icp_matches(xyz0, xyz1, max_dist=maximum_distance)
    all_matches, all_conf, all_candidate_matches = out
    matches_with_conf = [(m[0], m[1], c[0]) for m, c in zip(all_matches, all_conf)]

    return matches_with_conf, all_candidate_matches, xyz0


def calc_matches_using_2nn(all_propagated, n1_locs, max_dist=5.0):
    """
    Custom function to calculate matches based on 2 nearest neighbors
        DEPRECATED

    See: calc_bipartite_from_distance
    """

    # Build tree to query v1 neurons
    num_n, _, tree_n1 = build_neuron_tree(n1_locs, to_mirror=False)

    all_matches = []  # Without confidence
    all_conf = []
    all_candidate_matches = []
    nn_opt = {'radius': max_dist, 'max_nn': 1}
    conf_func = lambda dist: 1.0 / (dist / 10 + 1.0)
    for i, neuron in enumerate(np.array(all_propagated.points)):
        [k, two_neighbors, two_dist] = tree_n1.search_hybrid_vector_3d(neuron, **nn_opt)
        # For some reason this function seems to allow points that are too far away
        if k == 0 or (two_dist[0] > nn_opt['radius']):
            continue

        if k == 1:
            dist = two_dist[0]
            i_match = two_neighbors[0]
        else:
            if two_dist[0] / two_dist[1] > distance_ratio:
                dist = two_dist[1]
                i_match = two_neighbors[1]
            elif two_dist[1] / two_dist[0] > distance_ratio:
                dist = two_dist[1]
                i_match = two_neighbors[1]
            else:
                if verbose >= 2:
                    print(f"Neuron {i} has two close neighbors")
                continue

        if verbose >= 2:
            print(f"Found good match for neuron {i}")
        all_matches.append([i, i_match])
        all_conf.append(conf_func(dist))

        # todo: clean this up
        all_m = [(i, n, conf_func(dist)) for n, dist in zip(two_neighbors, two_dist)]
        all_candidate_matches.extend(all_m)

    return all_matches, all_conf, all_candidate_matches


##
## Visualization
##

def create_affine_visualizations(which_neuron, f0, f1, neuron0_trans):
    # Original neurons
    pc0_neuron = o3d.geometry.PointCloud()
    pc0_neuron.points = o3d.utility.Vector3dVector(f0.neuron_locs)
    pc0_neuron.paint_uniform_color([0.5, 0.5, 0.5])
    np.asarray(pc0_neuron.colors)[which_neuron, :] = [1, 0, 0]

    # Visualize the correspondence
    pc1_trans = o3d.geometry.PointCloud()
    pc1_trans.points = o3d.utility.Vector3dVector(neuron0_trans)
    pc1_trans.paint_uniform_color([0, 1, 0])

    corr = [(which_neuron, 0)]
    line = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pc0_neuron, pc1_trans, corr)

    # Visualize this neuron with the new neurons as well
    pc1_neuron = o3d.geometry.PointCloud()
    pc1_neuron.points = o3d.utility.Vector3dVector(f1.neuron_locs)
    pc1_neuron.paint_uniform_color([0, 0, 1])

    return pc0_neuron, pc1_trans, pc1_neuron, line
