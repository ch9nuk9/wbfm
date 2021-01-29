import open3d as o3d
import cv2
import numpy as np
from DLC_for_WBFM.utils.feature_detection.utils_features import build_feature_tree, build_neuron_tree
from tqdm import tqdm



def propagate_via_affine_model(which_neuron, f0, f1, all_feature_matches,
                               radius=10.0,
                               min_matches=100,
                               verbose=0):
    """
    1. Gets a cloud of features around a neuron (first frame)
    2. Fits an affine model based on the feature matches (second frame)
    3. Propagates the neuron to a final position
    """

    ## Get a neuron, then get the features around it
    this_neuron = f0.neuron_locs[which_neuron]

    num_features, pc0, tree_features0 = build_feature_tree(f0.keypoint_locs)
    pc0.paint_uniform_color([0.9,0.9,0.9])

    # See also calc_2frame_matches
    for i in range(5):
        nn_opt = {'radius':radius, 'max_nn':5000}
        [_, close_features, _] = tree_features0.search_hybrid_vector_3d(np.asarray(this_neuron), **nn_opt)

        ## Get the next-frame-matches of the features in this cloud

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
            break

    np.asarray(pc0.colors)[close_features[1:], :] = [0, 1, 0]

    # Now calculate the affine transformation
    neuron0_trans = None
    success = False
    if (len(pts0)>2) & (len(pts1)>2):
        val, h, inliers = cv2.estimateAffine3D(pts0,pts1, confidence=0.999)
        if verbose >= 2:
            print(f"Found {len(pts0)} matches from {len(close_features)} features")

        if h is not None:
            # And translate the neuron itself
            neuron_matrix = f0.neuron_locs[which_neuron:which_neuron+1]
            neuron0_trans = cv2.transform(np.array([neuron_matrix]), h)[0]

            success = True
    else:
        if verbose >= 2:
            print("Failed to find a match")

    return success, neuron0_trans


def propagate_all_neurons(f0, f1, all_feature_matches,
                          radius=10.0,
                          min_matches=100,
                          verbose=0):
    """
    Loops over neurons in f0 (frame 0), and applies:
        propagate_via_affine_model(which_neuron, f0, f1, all_feature_matches)
    """
    all_propagated = None

    opt = {'f0':f0, 'f1':f1, 'all_feature_matches':all_feature_matches}
    # for which_neuron in tqdm(range(len(f0.neuron_locs))):
    for which_neuron in range(len(f0.neuron_locs)):
        success, n0_propagated = propagate_via_affine_model(which_neuron, **opt)
        if not success:
            continue
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
                                          distance_ratio=1.5,
                                          verbose=0,
                                          DEBUG=False):
    """
    Propogates the neuron cloud in f0 using all_feature_matches
    Matches to neurons in f1 if the neighbors are close enough
    """

    all_propagated = propagate_all_neurons(f0, f1, all_feature_matches,
                              radius=radius,
                              min_matches=min_matches)

    # Build tree to query v1 neurons
    num_n, _, tree_n1 = build_neuron_tree(f1.neuron_locs, to_mirror=False)

    # Loop over locations of pushed v0 neurons
    all_matches = [] # Without confidence
    all_conf = []
    nn_opt = { 'radius':20.0, 'max_nn':2}
    conf_func = lambda dist : 1.0 / (dist/10+1.0)
    for i, neuron in enumerate(np.array(all_propagated.points)):
        [k, two_neighbors, two_dist] = tree_n1.search_hybrid_vector_3d(neuron, **nn_opt)

        if k==0:
            continue

        if k==1:
            dist = two_dist[0]
            i_match = two_neighbors[0]
        else:
            if two_dist[0]/two_dist[1] > distance_ratio:
                dist = two_dist[1]
                i_match = two_neighbors[1]
            elif two_dist[1]/two_dist[0] > distance_ratio:
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

    all_candidate_matches = None # For signature matching
    return all_matches, all_conf, all_candidate_matches

##
## Visualization
##

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
