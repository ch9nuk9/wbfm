from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import calc_bipartite_matches, build_digraph_from_matches, unpack_node_name
from networkx.algorithms.community import k_clique_communities
import networkx as nx
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


##
## Non networkx way to get bipartite matches
##

def calc_bipartite_from_distance(xyz0, xyz1, max_dist=None):
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
    # TODO: use sparse distance matrix: https://stackoverflow.com/questions/52366421/how-to-do-n-d-distance-and-nearest-neighbor-calculations-on-numpy-arrays
    cost_matrix = cdist(np.array(xyz0), np.array(xyz1), 'euclidean')
    matches = linear_sum_assignment(cost_matrix)
    matches = [[m0, m1] for (m0, m1) in zip(matches[0], matches[1])]

    # Postprocess to remove distance matches
    if max_dist is not None:
        match_dist = [cost_matrix[i,j] for (i,j) in matches]
        to_remove = [i for i, d in enumerate(match_dist) if d>max_dist]
        to_remove.reverse()
        [matches.pop(i) for i in to_remove]

        conf_func = lambda dist : 1.0 / (dist/max_dist+1.0)
    else:
        conf_func = lambda dist : 1.0 / (dist/10.0+1.0)

    # Calculate confidences from distance
    match_dist = [cost_matrix[i,j] for (i,j) in matches]
    conf = [conf_func(d) for d in match_dist]

    # Return matches twice to fit old function signature
    return matches, conf, matches


##
## Convinience function
##

def calc_all_bipartite_matches(candidates, min_edge_weight=0.5):
    bp_match_dict = {}
    for key in candidates:
        these_candidates = [c for c in candidates[key] if c[-1]>min_edge_weight]
        bp_matches = calc_bipartite_matches(these_candidates)
        bp_match_dict[key] = bp_matches

    return bp_match_dict


##
## Build communities from large network of matches
##

def calc_neurons_using_k_cliques(all_matches,
                                 k_values = [5,4,3],
                                 list_min_sizes = [450, 400, 350, 300, 250],
                                 max_size = 500,
                                 min_conf=0.0,
                                 verbose=1):
    # Do a list of descending clique sizes
    G = build_digraph_from_matches(all_matches, verbose=0, min_conf=min_conf).to_undirected()

    # Precompute cliques... doesn't work if nodes are removed
    # all_cliques = list(nx.find_cliques(G))

    all_communities = []
    # Multiple passes: take largest communities first
    for min_size in list_min_sizes:
        for k in k_values:
            communities = list(k_clique_communities(G, k=k))#, cliques=all_cliques))
            nodes_to_remove = []
            for c in communities:
                if len(c) > min_size and len(c) < max_size:
                    nodes_to_remove.extend(c)
                    all_communities.append(c)
            G.remove_nodes_from(nodes_to_remove)
            if verbose >= 1:
                print(f"{len(G.nodes)} nodes remaining")
        max_size = min_size

    return all_communities

##
## Utilities
##

##
## Networkx conversion
##

def convert_labels_to_matches(labels, offset=None, max_frames=None, DEBUG=False):
    """
    Turns a dict of classes per neuron (labels) into framewise matches
        Note: not every neuron needs to be labeled

    Assumes the node indices can be unpacked using unpack_node_name()
    """

    match_dict = defaultdict(list)
    unique_labels = np.unique(list(labels.values()))
    if DEBUG:
        print(unique_labels)

    for name in unique_labels:
        # Get nodes of this class
        these_ind = []
        for node_ind, lab in labels.items():
            if lab!=name:
                continue
            frame_ind, local_ind = unpack_node_name(node_ind)
            these_ind.append((frame_ind, local_ind))
        if DEBUG:
            print(these_ind)
        # Build matches that know the starting frame
        for i_f0, i_l0 in these_ind:
            for i_f1, i_l1 in these_ind:
                if i_l0 == i_l1 and i_f0==i_f1:
                    continue
                if offset is not None:
                    k = (i_f0-offset, i_f1-offset)
                else:
                    k = (i_f0, i_f1)
                if k[0]>=max_frames or k[1]>=max_frames:
                    continue
                match_dict[k].append([i_l0, i_l1])

    return match_dict


def community_to_matches(all_communities):
    """See calc_neurons_using_k_cliques()"""

    community_dict = {}
    for i, c in enumerate(all_communities):
        name = f"neuron_{i}"
        for neuron in c:
            community_dict[neuron]=name
    clique_matches = convert_labels_to_matches(community_dict, offset=50, max_frames=500)

    return clique_matches
