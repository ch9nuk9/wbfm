import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm
import pickle
from DLC_for_WBFM.utils.feature_detection.utils_features import build_neuron_tree
from scipy.spatial.distance import cdist
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import calc_bipartite_matches


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


def calc_feature_dist(f0, f1,
                      check_distance_early,
                      max_distance,
                      use_cdist=True):
    """Like a Hausdorf distance, but averaging instead of worst case"""

    if use_cdist:
        V01 = np.var(f0, axis=0, ddof=1) # Only do variance of f0
        all_pairwise_dist = cdist(f0,f1, metric='seuclidean', V=V01)
        f01_dist = np.min(all_pairwise_dist, axis=1)
        # CV01 = np.inv(np.cov(f0.T)).T # Only do variance of f0
        # all_pairwise_dist = cdist(f0,f1, metric='mahalnobis', CV=V01)
        # f01_dist = np.min(all_pairwise_dist, axis=1)
    else:
        sigma0 = np.std(f0,axis=0)
        f01_dist = []
        for t0 in range(f0.shape[0]):
            t0_dist = []
            v0 = f0[t0,:]/sigma0
            for t1 in range(f1.shape[0]):
                v1 = f1[t1,:]/sigma0
                t0_dist.append(np.linalg.norm(v1-v0))
                if t1==check_distance_early and (np.mean(t0_dist)>max_distance):
                    # Don't bother to calculate the rest of the distances
                    break
            f01_dist.append(np.min(t0_dist))

    return np.mean(f01_dist), f01_dist


##
## Manipulations of distance matrix
##

def distance_to_similarity_matrix(dist_mat, max_dist=10.0):
    """
    Build symmetrized similarity matrix from distance matrix
    Assumes that 'no connection' is np.nan in the distance matrix
    """
    d = dist_mat.copy()
    d[d>max_dist] = np.nan
    d = d + d.T # If the other direction is nan, now both are
    d = 1.0/d
    return d


##
## Final matching
##


def match_tracklets_using_features(all_tracklet_features,
                                   max_distance=15.0,
                                   check_distance_early=10,
                                   use_cdist=True,
                                   enforce_bidirectional_matches=True,
                                   check_index_overlap=True,
                                   tracklet_df=None,
                                   force_symmetry=False):
    """
    Matches tracklets using pairwise matching of frames within the tracklets

    tracklet_df is only needed if metadata is being checked
    """

    if check_index_overlap:
        assert not tracklet_df is None

    dist_opt = {'check_distance_early':check_distance_early,
                'max_distance':max_distance,
                'use_cdist':use_cdist}
    # Build distance matrix
    n = len(all_tracklet_features)
    all_dist = np.zeros((n,n))
    all_dist[:] = np.nan
    for i0 in tqdm(range(n)):
        f0 = np.array(all_tracklet_features[i0])
        if force_symmetry:
            i_start = i0+1
        else:
            i_start = 0
        for i1 in range(i_start, n):
            if i0 == i1:
                continue
            if check_index_overlap:
                if is_any_index_overlap(i0,i1,tracklet_df):
                    continue
            f1 = np.array(all_tracklet_features[i1])
            # Match this feature (min of partner)
            dist, _ = calc_feature_dist(f0, f1, **dist_opt)
            all_dist[i0,i1] = dist
            if force_symmetry:
                # FOR NOW: force symmetry
                all_dist[i1,i0] = dist

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

    # Postprocess matches
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
## Stepwise bipartite matching
##

def get_sources(df, i, min_len=3):
    """Get the indices of the nodes ending at the given frame"""
    f = lambda ind : (ind[-1]==i) and (len(ind)>min_len)
    ind = np.where(df['slice_ind'].apply(f))[0]
    clust_ind = df['clust_ind'].iloc[ind]
    return list(zip(ind, list(clust_ind)))


def get_sinks(df, i, min_len=3):
    """Get the indices of the nodes starting at the given frame"""
    f = lambda ind : (ind[0]==i) and (len(ind)>min_len)
    ind = np.where(df['slice_ind'].apply(f))[0]
    clust_ind = df['clust_ind'].iloc[ind]
    return list(zip(ind, list(clust_ind)))


def get_sink_source_matches(df, all_dist, i, source_ind=[]):
    """Assume that all_dist has been made symmetric

    Also assume that np.nan means no connection"""
    new_source_ind = get_sources(df, i-1)
    source_ind.extend(new_source_ind)
    sink_ind = get_sinks(df, i)

    all_candidates = []
    # These are each tuples; the first entry is the matrix index
    # i.e. for interfacing with the distance matrix
    for i0 in source_ind:
        for i1 in sink_ind:
            w = all_dist[i0[0],i1[0]]
            if not np.isnan(w):
                all_candidates.append([i0[1],i1[1],w])

    return all_candidates


def stepwise_bipartite_match(df, all_dist, i_start, num_frames):
    """Matches tracklets by optimizing over every step, and taking the first optimal match"""

    previous_sources = []
    all_bp_matches = []
    all_candidates = []
    for i in range(i_start, i_start+num_frames):
        candidates = get_sink_source_matches(df, all_dist, i, source_ind=previous_sources)
        bp_matches = calc_bipartite_matches(candidates)

        # Remove the found matches
        to_pop = []
        next_sources = []
        for s in previous_sources:
            for m in bp_matches:
                if s[1] == m[0]:
                    break
            else:
                # i.e. no match exists
                next_sources.append(s)
        previous_sources = next_sources

        all_bp_matches.extend(bp_matches)
        all_candidates.append(candidates)

    return all_bp_matches, all_candidates


##
## Utitlies
##

def get_indices_of_tracklet(i, tracklet_df):
    ind = tracklet_df['slice_ind'].iloc[i]
    return ind


def get_index_overlap(i0, i1, tracklet_df):
    ind0 = set(get_indices_of_tracklet(i0, tracklet_df))
    ind1 = set(get_indices_of_tracklet(i1, tracklet_df))
    return ind0.intersection(ind1)


def is_any_index_overlap(i0, i1, tracklet_df):
    overlap = get_index_overlap(i0, i1, tracklet_df)
    return len(overlap)>0


def get_index_overlap_list(all_i, verbose=1):
    n = len(all_i)
    for i0 in range(n):
        for i1 in range(i0+1,n):
            overlap = get_index_overlap(all_i[i0], all_i[i1])
            if len(overlap) > 1 or verbose >= 1:
                print(f"Overlap between {all_i[i0]} and {all_i[i1]}:")
                print(overlap)


def visualize_tracklet_in_body(i_tracklet, i_frame,
                               tracklet_df, kp_df, all_frames,
                               to_plot=False):
    if type(i_tracklet)!=list:
        i_tracklet = [i_tracklet]

    for i_t in i_tracklet:
        tracklet_ind = get_indices_of_tracklet(i_t, tracklet_df)
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
