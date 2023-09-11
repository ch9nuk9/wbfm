import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

from wbfm.utils.neuron_matching.class_reference_frame import ReferenceFrame
from wbfm.utils.neuron_matching.utils_matching import calc_nearest_neighbor_matches


def propagate_via_affine_model(which_neuron: int,
                               f0: ReferenceFrame,
                               f1: ReferenceFrame,
                               all_feature_matches: dict,
                               radius=10.0,
                               min_matches=100,
                               no_match_mode='negative_position',
                               allow_z_change=False,
                               verbose=0):
    """
    1. Gets a cloud of features around a neuron (first frame)
    2. Fits an affine model based on the feature matches (second frame)
    3. Propagates the neuron to a final position

    If the affine fitting fails, then a dummy point is added according to 'no_match_mode'
    The default is to add a point add [-10, -10, -10]
    (this keeps the indices of the pushed point cloud aligned with the original)
    """
    # global close_features, pts0, pts1
    pts0, pts1 = get_ball_of_points_to_use_for_local_affine(which_neuron, all_feature_matches, f0, f1, min_matches, radius)

    # Now calculate and apply the affine transformation
    if no_match_mode == 'negative_position':
        neuron0_trans = np.array([[-10.0, -10.0, -10.0]])
    else:
        neuron0_trans = None
    success = False
    if (len(pts0) > 2) & (len(pts1) > 2):
        val, h, inliers = cv2.estimateAffine3D(pts0, pts1, confidence=0.999)

        if h is not None:
            # And translate the neuron itself
            neuron_matrix = f0.neuron_locs[which_neuron:which_neuron + 1]
            neuron0_trans = cv2.transform(np.array([neuron_matrix]), h)[0]

            success = True

            if success and not allow_z_change:
                neuron0_trans[:, 0] = neuron_matrix[:, 0]
    else:
        if verbose >= 2:
            print("Failed to find a match")

    return success, neuron0_trans


def get_ball_of_points_to_use_for_local_affine(which_neuron, all_feature_matches,
                                               f0: ReferenceFrame, f1: ReferenceFrame, min_matches, radius):
    this_neuron = np.asarray(f0.neuron_locs[which_neuron])
    nbr_obj = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(f0.keypoint_locs)

    # Iteratively increases the radius if not enough matches are found
    for i in range(5):
        nn_opt = {'radius': radius, 'return_distance': False}
        close_features = nbr_obj.radius_neighbors(this_neuron.reshape(1, -1), **nn_opt)[0]

        # Get the next-frame-matches of the features in this cloud
        # Get just these points, and align two lists
        pts0, pts1 = [], []
        for match in all_feature_matches:
            try:
                # Already converted to list
                v0_ind = int(match[0])
            except AttributeError:
                v0_ind = match.queryIdx
            if v0_ind in close_features:
                pts0.append(f0.keypoint_locs[v0_ind])
                try:
                    pts1.append(f1.keypoint_locs[int(match[1])])
                except AttributeError:
                    pts1.append(f1.keypoint_locs[match.trainIdx])
        pts0, pts1 = np.array(pts0), np.array(pts1)

        if pts0.shape[0] < min_matches:
            radius = radius * 1.5
        else:
            break

    return pts0, pts1


def propagate_all_neurons(f0: ReferenceFrame, f1: ReferenceFrame, all_feature_matches,
                          radius=10.0,
                          min_matches=100,
                          allow_z_change=False,
                          verbose=0):
    """
    Loops over neurons in f0 (frame 0), and applies:
        propagate_via_affine_model(which_neuron, f0, f1, all_feature_matches)
    """
    all_propagated = np.zeros((len(f0.neuron_locs), 3))

    options = {'f0': f0, 'f1': f1, 'all_feature_matches': all_feature_matches,
               'radius': radius, 'min_matches': min_matches, 'allow_z_change': allow_z_change}
    for which_neuron in range(len(f0.neuron_locs)):
        success, n0_propagated = propagate_via_affine_model(which_neuron, **options)

        all_propagated[which_neuron, :] = n0_propagated

    return all_propagated


def calc_matches_using_affine_propagation(f0: ReferenceFrame, f1: ReferenceFrame,
                                          all_feature_matches,
                                          radius=10.0,
                                          min_matches=100,
                                          maximum_distance=15.0,
                                          allow_z_change=False,
                                          num_candidates=1,
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
                                           min_matches=min_matches,
                                           allow_z_change=allow_z_change)

    xyz0 = all_propagated
    xyz1 = f1.neuron_locs

    out = calc_nearest_neighbor_matches(xyz0, xyz1, max_dist=maximum_distance, n_neighbors=num_candidates)
    all_matches, all_conf = out
    if len(all_matches) == 0:
        return [], [], []
    all_candidate_matches = all_matches
    assert np.isscalar(all_conf[0]), "Check formatting (nested lists)"
    matches_with_conf = [(m[0], m[1], c) for m, c in zip(all_matches, all_conf)]

    return matches_with_conf, all_candidate_matches, xyz0
