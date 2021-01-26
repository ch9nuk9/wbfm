import open3d as o3d
import cv2
import numpy as np
from DLC_for_WBFM.utils.feature_detection.utils_features import build_feature_tree




def match_via_affine_model(which_neuron, f0, f1, verbose=0):

    ## Get a neuron, then get the features around it
    this_neuron = f0.neuron_locs[which_neuron]

    num_features, pc0, tree_features0 = build_feature_tree(f0.keypoint_locs)
    pc0.paint_uniform_color([0.9,0.9,0.9])

    # See also calc_2frame_matches
    has_enough_matches = False
    min_matches = 100
    radius = 20.0
    while not has_enough_matches:
        nn_opt = {'radius':radius, 'max_nn':5000}
        [_, close_features, _] = tree_features0.search_hybrid_vector_3d(np.asarray(this_neuron), **nn_opt)

        ## Get the next-frame-matches of the features in this cloud

        # Pre-calculated matches for all keypoints
        all_feature_matches = reference_set.feature_matches[(0,1)]

        # Get just these points, and align two lists
        pts0, pts1 = [], []
        for match in all_feature_matches:
            v0_ind = match.queryIdx
            if v0_ind in close_features:
                pts0.append(f0.keypoint_locs[v0_ind])
                pts1.append(f1.keypoint_locs[match.trainIdx])
        pts0, pts1 = np.array(pts0), np.array(pts1)

        if len(pts0) < min_matches:
            radius = radius*1.5
        else:
            has_enough_matches = True

    np.asarray(pc0.colors)[close_features[1:], :] = [0, 1, 0]

    # Now calculate the affine transformation
    if (len(pts0)>2) & (len(pts1)>2):
        val, h, inliers = cv2.estimateAffine3D(pts0,pts1, confidence=0.99)
        if verbose >= 2:
            print(f"Found {len(pts0)} matches from {len(close_features)} features")

        # And translate the neuron itself
        neuron_matrix = f0.neuron_locs[which_neuron:which_neuron+1]
        neuron0_trans = cv2.transform(np.array([neuron_matrix]), h)[0]

        success = True
    else:
        if verbose >= 2:
            print("Failed to find a match")
        success = False
        neuron0_trans = None

    return success, neuron0_trans


def create_affine_visualizations(which_neuron, f0, f1, neuron0_trans):
    # Original neurons
    pc0_neuron = o3d.geometry.PointCloud()
    pc0_neuron.points = o3d.utility.Vector3dVector(f0.neuron_locs)
    pc0_neuron.paint_uniform_color([0.5,0.5,0.5])
    np.asarray(pc0_neuron.colors)[which_neuron, :] = [1, 0, 0]

    # Visualize the correspondence
    pc1_trans = o3d.geometry.PointCloud()
    pc1_trans.points = o3d.utility.Vector3dVector(neuron0_trans)
    pc1_trans.paint_uniform_color([0,1,0])

    corr = [(which_neuron,0)]
    line = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pc0_neuron, pc1_trans, corr)

    # Visualize this neuron with the new neurons as well
    pc1_neuron = o3d.geometry.PointCloud()
    pc1_neuron.points = o3d.utility.Vector3dVector(f1.neuron_locs)
    pc1_neuron.paint_uniform_color([0,0,1])

    return pc0_neuron, pc1_trans, pc1_neuron, line
