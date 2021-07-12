from typing import Tuple, List
import open3d as o3d
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import networkx as nx


##
## Use networkx to do bipartite matching
##

def calc_bipartite_matches(all_candidate_matches, verbose=0):
    """
    Calculates the globally optimally matching from an overmatched array with weights

    Uses nx.max_weight_matching()
        i.e. assumes weight=good, for example confidence
    """

    G = nx.Graph()
    # Rename the second frame's neurons so the graph is truly bipartite
    for candidate in all_candidate_matches:
        candidate = list(candidate)
        candidate[1] = get_node_name(1, candidate[1])
        #candidate[2] = 1/candidate[2]
        # Otherwise the sets are unordered
        G.add_node(candidate[0], bipartite=0)
        G.add_node(candidate[1], bipartite=1)
        # Default weight
        if len(candidate)==2:
            candidate.append(1)
        G.add_weighted_edges_from([candidate])
    if verbose >= 2:
        print("Performing bipartite matching")
    #set0 = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    tmp_bp_matches = nx.max_weight_matching(G, maxcardinality=True)
    #all_bp_dict = nx.bipartite.minimum_weight_full_matching(G, set0)
    # Translate back into neuron index space
    # all_bp_matches = []
    # for neur0,v in all_bp_dict.items():
    #     neur1 = unpack_node_name(v)[1]
    #     all_bp_matches.append([neur0, neur1])
    all_bp_matches = []
    for m in tmp_bp_matches:
        m = list(m) # unordered by default
        m.sort()
        m[1] = unpack_node_name(m[1])[1]
        all_bp_matches.append(m)

    return all_bp_matches


##
## General network utilities
##

def get_node_name(frame_ind, neuron_ind):
    """The graph is indexed by integer, so all neurons must be unique"""
    return frame_ind*10000 + neuron_ind

def unpack_node_name(node_name):
    """Inverse of get_node_name"""
    # if np.issubdtype(type(node_name), np.integer):
    try:
        return divmod(node_name, 10000)
    except:
        if type(node_name)==tuple:
            return node_name
        else:
            print("Must pass integer or, trivially, a tuple")
            raise ValueError



def build_digraph_from_matches(pairwise_matches,
                               pairwise_conf=None,
                               min_conf=0.0,
                               verbose=1):
    DG = nx.DiGraph()
    for frames, all_neurons in pairwise_matches.items():
        if verbose >= 1:
            print("==============================")
            print("Analyzing pair:")
            print(frames)
        if pairwise_conf is not None:
            all_conf = pairwise_conf[frames]
        else:
            all_conf = np.ones_like(np.array(all_neurons)[:,0])
        for neuron_pair, this_conf in zip(all_neurons, all_conf):
            if this_conf < min_conf:
                continue
            node1 = get_node_name(frames[0], neuron_pair[0])
            node2 = get_node_name(frames[1], neuron_pair[1])
            e = (node1, node2, this_conf)
            DG.add_weighted_edges_from([e])

    return DG




##
## Alternate, non-networkx way to get bipartite matches
##


def calc_bipartite_from_candidates(all_candidate_matches, gamma=1.0, min_conf=1e-3, verbose=0):
    """
    Sparse version of calc_bipartite_from_distance

    starts from a list of matches (may repeat, but are not full) with confidences

    Uses scipy linear_sum_assignment
    Note: does not use scipy.sparse.csgraph.min_weight_full_bipartite_matching for version compatibility
    """
    # OPTIMIZE: this produces a larger matrix than necessary

    m0 = np.max([m[0] for m in all_candidate_matches]) + 1
    m1 = np.max([m[1] for m in all_candidate_matches]) + 1
    # largest_neuron_ind = np.max([max([m[0], m[1]]) for m in all_candidate_matches]) + 1
    # sz = (m0, largest_neuron_ind)
    conf_matrix = np.zeros((m0, m1))
    for i0, i1, conf in all_candidate_matches:
        conf_matrix[i0, i1] += conf

    matches = linear_sum_assignment(conf_matrix, maximize=True)
    matches = [[m0, m1] for (m0, m1) in zip(matches[0], matches[1])]
    # Apply sigmoid to summed confidence
    matches = np.array(matches)
    conf = np.array([np.tanh(conf_matrix[i0, i1]) for i0, i1 in matches])

    to_keep = conf > min_conf
    matches = matches[to_keep]
    conf = conf[to_keep]

    return matches, conf, matches


def calc_bipartite_from_distance(xyz0: np.ndarray, xyz1: np.ndarray,
                                 max_dist: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses scipy implementation of linear_sum_assignment to calculate best matches

    Parameters
    ==============
    xyz0 - array-like; shape=(n0,m)
        The 3d positions of a point cloud
        Note that m==3 is not required
    xyz1 - array-like; shape=(n1,m)
        The 3d positions of a second point cloud
    max_dist - float or None (default)
        Distance over which to remove matches

    """
    # ENHANCE: use sparse distance matrix: https://stackoverflow.com/questions/52366421/how-to-do-n-d-distance-and-nearest-neighbor-calculations-on-numpy-arrays
    cost_matrix = cdist(np.array(xyz0), np.array(xyz1), 'euclidean')
    matches = linear_sum_assignment(cost_matrix)
    raw_matches = [[m0, m1] for (m0, m1) in zip(matches[0], matches[1])]
    matches = raw_matches.copy()

    # Postprocess to remove distance matches
    if max_dist is not None:
        match_dist = [cost_matrix[i, j] for (i, j) in matches]
        to_remove = [i for i, d in enumerate(match_dist) if d > max_dist]
        to_remove.reverse()
        [matches.pop(i) for i in to_remove]

    matches = np.array(matches)
    conf = calc_confidence_from_distance_array_and_matches(cost_matrix, matches)
    # conf = [conf_func(d) for d in match_dist]

    # Return matches twice to fit old function signature
    return matches, conf, np.array(raw_matches)


def calc_confidence_from_distance_array_and_matches(distance_matrix, matches):
    conf_func = lambda dist: np.tanh(1.0 / dist)
    # Calculate confidences from distance
    conf = np.zeros((matches.shape[0], 1))
    for i, (m0, m1) in enumerate(matches):
        dist = distance_matrix[m0, m1]
        conf[i] = conf_func(dist)
    return conf


def calc_icp_matches(xyz0: np.ndarray, xyz1: np.ndarray,
                     max_dist: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates matches between lists of 3d points (which may have outliers) using ICP

    Currently using open3d implementation of ICP
    TODO: does this work on a cluster?

    Parameters
    ----------
    xyz0
    xyz1
    max_dist

    Returns
    -------

    See also: calc_bipartite_from_distance
        (uses same API)

    """
    zxy0, zxy1 = np.array(xyz0), np.array(xyz1)

    pc0 = o3d.geometry.PointCloud()
    if len(xyz0) > 0:
        pc0.points = o3d.utility.Vector3dVector(xyz0)
    else:
        return np.array([]), np.array([]), np.array([])

    pc1 = o3d.geometry.PointCloud()
    if len(xyz1) > 0:
        pc1.points = o3d.utility.Vector3dVector(xyz1)
    else:
        return np.array([]), np.array([]), np.array([])

    # Do greedy matching
    icp_result = o3d.pipelines.registration.registration_icp(pc0, pc1, max_correspondence_distance=max_dist)
    matches = np.array(icp_result.correspondence_set)

    # Calculate confidences
    dist_matrix = cdist(zxy0, zxy1, 'euclidean')
    conf = calc_confidence_from_distance_array_and_matches(dist_matrix, matches)

    return matches, conf, matches


def is_one_neuron_per_frame(node_names, min_size=None, total_frames=None):
    """
    Checks a connected component (list of nodes) to make sure each frame is only represented once
    """

    # Heuristic check
    sz = len(node_names)
    if total_frames is not None and sz > total_frames:
        return False
    if min_size is not None and sz < min_size:
        return False

    # Actual check for duplicates
    all_frames = [unpack_node_name(n)[0] for n in node_names]

    if len(all_frames) > len(set(all_frames)):
        return False
    return True
