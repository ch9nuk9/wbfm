from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_matches, build_digraph_from_matches, unpack_node_name, is_one_neuron_per_frame
from networkx.algorithms.community import k_clique_communities
import networkx as nx
from collections import defaultdict
import numpy as np

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


def calc_neuron_using_voronoi(all_matches,
                              dist,
                              total_frames,
                              target_size_vec=None,
                              verbose=0):
    # Cluster using voronoi cells
    DG = build_digraph_from_matches(all_matches, dist, verbose=0)
    # Indices may not start at 0
    # Syntax: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    all_pairs = all_matches.keys()
    unique_nodes = set(node for pair in all_pairs for node in pair)

    global2local = {}
    global_current_ind = 0
    if target_size_vec is None:
        stop_size = max(1, int(total_frames/2))
        target_size_vec = list(range(total_frames,stop_size,-1))

    for target_size in target_size_vec:
        for start_vol in unique_nodes:
            # Get simple centers: all neurons in a "start" volume
            center_nodes = []
            for n in DG.nodes():
                frame_ind, _ = unpack_node_name(n)
                if frame_ind == start_vol:
                    center_nodes.append(n)
            if len(center_nodes)==0:
                continue
            cells = nx.voronoi_cells(DG, center_nodes)

            # Heuristic
            # If the cells have a unique node in each frame, then take it as true
            # TODO: removal of outliers
            for k, v in cells.items():
                if is_one_neuron_per_frame(v, min_size=target_size, total_frames=total_frames):
                    global2local[global_current_ind] = v
                    global_current_ind += 1
                    DG.remove_nodes_from(v)
            if verbose >= 2:
                print(f"{len(DG)} nodes remaining (across all frames)")
        if verbose >= 1:
            print(f"Found {global_current_ind} neurons of size at least {target_size}")

    return global2local

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
    Format:
        labels[node_ind] = global_neuron_label

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
                if max_frames is not None:
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
