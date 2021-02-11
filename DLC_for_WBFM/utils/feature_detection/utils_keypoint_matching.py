import numpy as np
import open3d as o3d
import pandas as pd

import pickle


##
## Parsing output of original tracker
##

# Getting keypoints
def separate_keypoints_and_tracklets(clust_df, start_ind, window_length,
                                     min_tracklet_overlap=3,
                                     min_required_keypoints=10):
    """
    For a given window, separates tracklets into two categories:
        Keypoints, which are tracked in the entire window
        Tracklets, which have at least 'min_tracklet_overlap' frames in the window
    """

    is_keypoint = []
    is_tracklet = []
    this_window = list(range(start_ind, start_ind + window_length))
    for i, row in enumerate(clust_df['slice_ind']):
        overlap_ind = np.array([i in this_window for i in row])
        num_overlap = np.count_nonzero(overlap_ind)
        if num_overlap < min_tracklet_overlap:
            continue
        elif num_overlap == window_length:
            is_keypoint.append(i)
        elif num_overlap <= (window_length-min_tracklet_overlap):
            # If it is nearly the entire window, then it can't match with anything
            is_tracklet.append(i)

    kp_df = clust_df.iloc[is_keypoint]
    tracklet_df = clust_df.iloc[is_tracklet]

    if len(kp_df) < min_required_keypoints:
        print(f"Warning, few keypoints ({len(kp_df)}) detected")

    return kp_df, tracklet_df


##
## Building the distance features
##

def calc_vectors(this_neuron, kp_df, slice_ind):
    one_neuron_vectors = []
    this_neuron = np.array(this_neuron)
    for _, kp in kp_df.iterrows():
        # The tracks are long, so only do the point at slice_ind
        kp_ind = kp['slice_ind'].index(slice_ind)
        kp_neuron = np.array(kp['all_xyz'][kp_ind])
        one_neuron_vectors.append(this_neuron - kp_neuron)

    return one_neuron_vectors


def calc_features_from_vectors(one_neuron_vectors, only_diagonal):
    """Todo: refactor to be smarter with inner product"""
    n = len(one_neuron_vectors)
    one_neuron_features = []
    for i1 in range(n):
        for i2 in range(i1, n):
            v1, v2 = one_neuron_vectors[i1], one_neuron_vectors[i2]
            one_neuron_features.append(np.dot(v1, v2))
            if only_diagonal:
                break

    return one_neuron_features


def calc_all_tracklet_features(kp_df, tracklet_df, start_ind, window_length, only_diagonal=True, verbose=1):
    """
    Processes tracklet dataframe, returning a list of features

    The indices of this list correspond to the overlap with the window
    """

    this_window = list(range(start_ind, start_ind + window_length))
    all_tracklet_features = []
    if verbose >= 1:
        print("Calculating features for all tracklets")
    for _, tracklet in tqdm(tracklet_df.iterrows(), total=len(tracklet_df)):
        this_tracklet_features = []
        for global_ind in this_window:
            if global_ind in tracklet['slice_ind']:
                tracklet_ind = tracklet['slice_ind'].index(global_ind)
                this_neuron = tracklet['all_xyz'][tracklet_ind]
            else:
                continue
            these_vectors = calc_vectors(this_neuron, kp_df, global_ind)
            tmp = calc_features_from_vectors(these_vectors, only_diagonal)
            this_tracklet_features.append(tmp)
        all_tracklet_features.append(this_tracklet_features)

    return all_tracklet_features

##
## Final matching
##


def match_tracklets_using_features(all_tracklet_features,
                                   max_distance=15.0,
                                   enforce_bidirectional_matches=True):
    """
    Matches tracklets using pairwise matching of frames within the tracklets

    """

    # Build distance matrix
    n = len(all_tracklet_features)
    all_dist = np.zeros((n,n))
    all_dist[:] = np.nan
    for i0 in tqdm(range(n)):
        f0 = np.array(all_tracklet_features[i0])
        sigma0 = np.std(f0,axis=0)
        for i1 in range(i0+1, n):
            f1 = np.array(all_tracklet_features[i1])
            # Match this feature (min of partner)
            f01_dist = []
            for t0 in range(f0.shape[0]):
                t0_dist = []
                v0 = f0[t0,:]/sigma0
                for t1 in range(f1.shape[0]):
                    v1 = f1[t1,:]/sigma0
                    t0_dist.append(np.linalg.norm(v1-v0))
                f01_dist.append(np.min(t0_dist))
            all_dist[i0,i1] = np.mean(f01_dist)
            # FOR NOW: force symmetry
            all_dist[i1,i0] = np.mean(f01_dist)

    # Build greedy matches
    edges = {}
    for i_row in range(all_dist.shape[0]):
        r = all_dist[i_row,:]
        try:
            dist = np.nanmin(r)
            if dist < max_distance:
                edges[(i_row, np.nanargmin(r))] = dist
        except:
            pass

    if enforce_bidirectional_matches:
        to_remove = []
        for k, v in edges.items():
            opposite = (k[1], k[0])
            if not opposite in edges:
                to_remove.append(k)
        for k in to_remove:
            del edges[k]

    return edges, all_dist


##
## Utitlies
##

def get_indices_of_tracklet(i):
    ind = tracklet_df['slice_ind'].iloc[i]
    return ind

def get_index_overlap(i0, i1):
    ind0 = set(get_indices_of_tracklet(i0))
    ind1 = set(get_indices_of_tracklet(i1))
    return ind0.intersection(ind1)

def get_index_overlap_list(all_i, verbose=1):
    n = len(all_i)
    for i0 in range(n):
        for i1 in range(i0+1,n):
            overlap = get_index_overlap(all_i[i0], all_i[i1])
            if len(overlap) > 1 or verbose >= 1:
                print(f"Overlap between {all_i[i0]} and {all_i[i1]}:")
                print(overlap)


def visualize_tracklet_in_body(i_tracklet, i_frame, to_plot=False):
    if type(i_tracklet)!=list:
        i_tracklet = [i_tracklet]

    for i_t in i_tracklet:
        tracklet_ind = get_indices_of_tracklet(i_t)
        if not i_frame in tracklet_ind:
            print(f"{i_frame} is not in tracklet; try one of {tracklet_ind}")

    # Get this tracklet
    tracklet_xyz = []
    for i_t in i_tracklet:
        try:
            local_ind = tracklet_df['slice_ind'].iloc[i_t].index(i_frame)
            tracklet_xyz.append(tracklet_df['all_xyz'].iloc[i_t][local_ind])
        except:
            print(f"{i_frame} not in tracklet {i_t}")
            if to_error:
                raise ValueError

    # Get keypoints
    kp_xyz = []
    for i_kp in range(len(kp_df)):
        local_ind = kp_df['slice_ind'].iloc[i_kp].index(i_frame)
        kp_xyz.append(kp_df['all_xyz'].iloc[i_kp][local_ind])

    # Get all other neurons
    this_frame = all_frames[i_frame]
    _, pc_neurons, pc_tree = build_neuron_tree(this_frame.neuron_locs, False)
    pc_neurons.paint_uniform_color([0.5,0.5,0.5])

    # Color the tracklet and keypoint neurons
    for xyz in tracklet_xyz:
        [k,idx,_] = pc_tree.search_knn_vector_3d(xyz, 1)
        np.asarray(pc_neurons.colors)[idx[:], :] = [1, 0, 0]
    for xyz in kp_xyz:
        [k,idx,_] = pc_tree.search_knn_vector_3d(xyz, 1)
        np.asarray(pc_neurons.colors)[idx[:], :] = [0, 0, 1]

    if to_plot:
        o3d.visualization.draw_geometries([pc_neurons])

    return pc_neurons
