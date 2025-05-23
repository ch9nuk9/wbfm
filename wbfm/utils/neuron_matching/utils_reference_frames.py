import collections

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from wbfm.utils.neuron_matching.class_reference_frame import RegisteredReferenceFrames
from wbfm.utils.general.utils_features import add_neuron_match
from wbfm.utils.general.utils_networkx import unpack_node_name, is_one_neuron_per_frame


##
## Utilities for combining frames into a reference set
##


def remove_first_frame(reference_set):
    # Remove first element
    new_frames = reference_set.reference_frames.copy()
    new_frames.pop()

    g2l = reference_set.global2local.copy()
    for key in g2l:
        g2l[key].pop()
    # Offset keys by one
    l2g = {}
    for key, val in reference_set.local2global.items():
        if key[0] == 0:
            continue
        new_key = (key[0] - 1, key[1])
        l2g[new_key] = val

    # Offset keys by one (both indices)
    pm, pc, fm = {}, {}, {}
    for key in reference_set.pairwise_matches:
        if key[0] == 0 or key[1] == 0:
            continue
        new_key = (key[0] - 1, key[1] - 1)
        pm[new_key] = reference_set.pairwise_matches[key]
        pc[new_key] = reference_set.pairwise_conf[key]
        fm[new_key] = reference_set.feature_matches[key]

    # Build a class to store all the information
    reference_set_minus1 = RegisteredReferenceFrames(
        g2l,
        l2g,
        new_frames,
        pm,
        pc,
        fm,
        None
    )

    return reference_set_minus1


##
## Related helper and visualization functions
##

def get_subgraph_with_strong_weights(DG, min_weight):
    # G = nx.from_numpy_matrix(DG, parallel_edges=False)
    G = DG.copy()
    edge_weights = nx.get_edge_attributes(G, 'weight')
    G.remove_edges_from((e for e, w in edge_weights.items() if w < min_weight))
    return G


def calc_connected_components(DG, only_strong_components=True):
    if only_strong_components:
        all_neurons = list(nx.strongly_connected_components(DG))
    else:
        all_neurons = list(nx.weakly_connected_components(DG))
    all_len = [len(c) for c in all_neurons]
    # print(all_len)
    big_comp = np.argmax(all_len)
    print("Largest connected component size: ", max(all_len))
    # print(big_comp)
    big_DG = DG.subgraph(all_neurons[big_comp])

    return big_DG, all_len, all_neurons


def plot_degree_hist(DG):
    degree_sequence = sorted([d for n, d in DG.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")


##
## For determining the full reference set
##


def add_all_good_components(G,
                            thresh=0.0,
                            reference_ind=0,
                            total_frames=10):
    """
    Loops through a list of all connected components and adds to a global dict

    Builds two dictionaries:
        Indexed by global neuron index, returning a list of local frame indices
        Indexed by frame index and local neuron index, returning the global ind

    Also removes these found neurons from the original graph, G
    """
    global2local = {}
    local2global = {}
    G_tmp = get_subgraph_with_strong_weights(G, thresh)
    all_components = list(nx.strongly_connected_components(G_tmp))
    for comp in all_components:
        is_good_cluster = is_one_neuron_per_frame(comp, total_frames=total_frames)
        if is_good_cluster:
            all_local_ind = []
            for global_ind in comp:
                frame_ind, local_ind = unpack_node_name(global_ind)
                local2global[(frame_ind, local_ind)] = reference_ind
                all_local_ind.append(local_ind)
            global2local[reference_ind] = all_local_ind
            reference_ind += 1
            G.remove_nodes_from(comp)

    return global2local, local2global, reference_ind, G


def is_ordered_subset(list1, list2):
    n = min(len(list1), len(list2))
    # First element must match
    for i in range(n):
        if list1[i] != list2[i]:
            return False
    return True


def calc_matches_using_feature_voting(frame0, frame1,
                                      feature_matches_dict,
                                      verbose=0,
                                      min_features_needed=2,
                                      DEBUG=False):
    """Basic matching using opencv features

    See also:
        calc_matches_using_gaussian_process
        calc_matches_using_affine_propagation
    """

    all_neuron_matches = []
    all_confidences = []
    all_candidate_matches = []
    for neuron0_ind, neuron0_loc in enumerate(frame0.iter_neurons()):
        # Get features of this neuron
        this_f0 = frame0.get_features_of_neuron(neuron0_ind)
        if DEBUG:
            print(f"=======Neuron {neuron0_ind}=========")
        # Use matches to translate to the indices of frame1
        this_f1 = []
        for f0 in this_f0:
            i_match = feature_matches_dict.get(f0)
            if i_match is not None:
                this_f1.append(i_match)
        if DEBUG:
            print("Features in volume 1: ", this_f1)
        # Get the corresponding neurons in vol1, and vote
        f2n = frame1.features_to_neurons
        this_n1 = [f2n.get(f1) for f1 in this_f1 if f1 in f2n]
        if DEBUG:
            print("Matching neuron in volume 1: ", this_n1)

        all_neuron_matches, all_confidences, all_candidate_matches = add_neuron_match(
            all_neuron_matches,
            all_confidences,
            neuron0_ind,
            min_features_needed,
            this_n1,
            verbose=verbose - 1,
            all_candidate_matches=all_candidate_matches
        )
    matches_with_conf = [(m[0], m[1], c) for m, c in zip(all_neuron_matches, all_confidences)]
    # Empty list to match other signatures
    return matches_with_conf, all_candidate_matches, []
