from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import calc_bipartite_matches, build_digraph_from_matches, unpack_node_name
from networkx.algorithms.community import k_clique_communities
import networkx as nx
from collections import defaultdict
import numpy as np




def calc_all_bipartite_matches(min_edge_weight=0.5):
    bp_match_dict = {}
    for key in all_matches:
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
                                 max_size = 500):
    # Do a list of descending clique sizes
    G = build_digraph_from_matches(all_matches, verbose=0).to_undirected()

    all_communities = []
    # Multiple passes: take largest communities first
    for min_size in list_min_sizes:
        for k in k_values:
            communities = list(k_clique_communities(G, k=k))
            nodes_to_remove = []
            for c in communities:
                if len(c) > min_size and len(c) < max_size:
                    nodes_to_remove.extend(c)
                    all_communities.append(c)
            G.remove_nodes_from(nodes_to_remove)
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
