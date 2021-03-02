from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.feature_detection.utils_features import *
from DLC_for_WBFM.utils.feature_detection.utils_affine import calc_matches_using_affine_propagation
from DLC_for_WBFM.utils.feature_detection.utils_rigid_alignment import align_stack, filter_stack
from DLC_for_WBFM.utils.feature_detection.utils_detection import *
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import *
from DLC_for_WBFM.utils.feature_detection.utils_gaussian_process import calc_matches_using_gaussian_process
import numpy as np
import networkx as nx
import collections
from dataclasses import dataclass
import scipy.ndimage as ndi
from collections import defaultdict


##
## Main convinience constructors
##

def build_reference_frame(dat_raw,
                          num_slices,
                          neuron_feature_radius,
                          preprocessing_settings=PreprocessingSettings(),
                          start_slice=2,
                          metadata={},
                          verbose=0):
    """Main convinience constructor for ReferenceFrame class"""
    dat = perform_preprocessing(dat_raw, preprocessing_settings)

    # Get neurons and features, and a map between them
    neuron_locs, _, _, icp_kps = detect_neurons_using_ICP(dat,
                                                         num_slices=num_slices,
                                                         alpha=1.0,
                                                         min_detections=3,
                                                         start_slice=start_slice,
                                                         verbose=0)
    neuron_locs = np.array([n for n in neuron_locs])
    if len(neuron_locs)==0:
        print("No neurons detected... check data settings")
        raise ValueError
    feature_opt = {'num_features_per_plane':1000, 'start_plane':5}
    kps, kp_3d_locs, features = build_features_1volume(dat, **feature_opt)

    # The map requires some open3d subfunctions
    num_f, pc_f, _ = build_feature_tree(kp_3d_locs, which_slice=None)
    _, _, tree_neurons = build_neuron_tree(neuron_locs, to_mirror=False)
    f2n_map = build_f2n_map(kp_3d_locs,
                           num_f,
                           pc_f,
                           neuron_feature_radius,
                           tree_neurons,
                           verbose=verbose-1)

    # Finally, my summary class
    f = ReferenceFrame(neuron_locs, kps, kp_3d_locs, features, f2n_map,
                       **metadata,
                       preprocessing_settings=preprocessing_settings)
    return f


def perform_preprocessing(dat_raw, preprocessing_settings:PreprocessingSettings):
    """
    Performs all preprocessing as set by the fields of preprocessing_settings

    See PreprocessingSettings for options
    """

    s = preprocessing_settings

    if s.do_filtering:
        dat_raw = filter_stack(dat_raw, s.filter_opt)

    if s.do_rigid_alignment:
        dat_raw = align_stack(dat_raw)

    if s.do_mini_max_projection:
        mini_max_size = s.mini_max_size
        dat_raw = ndi.maximum_filter(dat_raw, size=(mini_max_size,1,1))

    dat_raw = (dat_raw*s.alpha).astype(s.final_dtype)

    return dat_raw


##
## Utilities for combining frames into a reference set
##

def get_node_name(frame_ind, neuron_ind):
    """The graph is indexed by integer, so all neurons must be unique"""
    return frame_ind*10000 + neuron_ind

def unpack_node_name(node_name):
    """Inverse of get_node_name"""
    return divmod(node_name, 10000)


def build_digraph_from_matches(pairwise_matches, pairwise_conf=None,
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
            #print(neuron_pair)
            node1 = get_node_name(frames[0], neuron_pair[0])
            node2 = get_node_name(frames[1], neuron_pair[1])
            e = (node1, node2, this_conf)
            DG.add_weighted_edges_from([e])

    return DG


def remove_first_frame(reference_set):
    # Remove first element
    new_frames = reference_set.reference_frames.copy()
    new_frames.pop()

    g2l = reference_set.global2local.copy()
    for key in g2l:
        g2l[key].pop()
    # Offset keys by one
    l2g = {}
    for key,val in reference_set.local2global.items():
        if key[0] == 0:
            continue
        new_key = (key[0]-1,key[1])
        l2g[new_key] = val

    # Offset keys by one (both indices)
    pm, pc, fm = {}, {}, {}
    for key in reference_set.pairwise_matches:
        if key[0]==0 or key[1]==0:
            continue
        new_key = (key[0]-1, key[1]-1)
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
    #G = nx.from_numpy_matrix(DG, parallel_edges=False)
    G = DG.copy()
    edge_weights = nx.get_edge_attributes(G,'weight')
    G.remove_edges_from((e for e, w in edge_weights.items() if w < min_weight))
    return G


def calc_connected_components(DG, only_strong_components=True):
    if only_strong_components:
        all_neurons = list(nx.strongly_connected_components(DG))
    else:
        all_neurons = list(nx.weakly_connected_components(DG))
    all_len = [len(c) for c in all_neurons]
    #print(all_len)
    big_comp = np.argmax(all_len)
    print("Largest connected component size: ", max(all_len))
    #print(big_comp)
    big_DG = DG.subgraph(all_neurons[big_comp])

    return big_DG, all_len, all_neurons


def plot_degree_hist(DG):
    degree_sequence = sorted([d for n, d in DG.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")


def calc_bipartite_matches(all_candidate_matches, verbose=0):
    """
    Calculates the globally optimally matching from an overmatched array with weights

    Parameters
    ==================
    all_candidate_matches : list of lists
        For example: [[0,1,0.1], [0,2,0.8]]
        Many candidate matches for neurons between two slices
        Assumes that the node in index [1] are local names
            Made to be used with get_node_name() and unpack_node_name()
        The value in index [2] is a weight; no need to be between 0.0 and 1.0


    Returns:
    =================
    all_bp_matches : list of lists
        For example: [[0,1], [1,2]]
        Same format as the input candidate matches, but WITHOUT weight
        But now are unique one-to-one matches
    """

    G = nx.Graph()
    # Rename the second frame's neurons so the graph is truly bipartite
    for candidate in all_candidate_matches:
        candidate = list(candidate)
        candidate[1] = get_node_name(1, candidate[1])
        # Otherwise the sets are unordered
        G.add_node(candidate[0], bipartite=0)
        G.add_node(candidate[1], bipartite=1)
        if len(candidate)==2:
            candidate.append(1)
        G.add_weighted_edges_from([candidate])
    if verbose >= 2:
        print("Performing bipartite matching")
    tmp_bp_matches = nx.max_weight_matching(G, maxcardinality=True)
    all_bp_matches = []
    for m in tmp_bp_matches:
        m = list(m) # unordered by default
        m.sort()
        m[1] = unpack_node_name(m[1])[1]
        all_bp_matches.append(m)

    return all_bp_matches


##
## For determining the full reference set
##


def is_one_neuron_per_frame(node_names, min_size=None, total_frames=10):
    """
    Checks a connected component (list of nodes) to make sure each frame is only represented once
    """
    if min_size is None:
        min_size = total_frames / 2.0

    # Heuristic check
    sz = len(node_names)
    if sz <= min_size or sz > total_frames:
        return False

    # Actual check
    all_frames = []
    for n in node_names:
        all_frames.append(unpack_node_name(n)[0])

    if len(all_frames) > len(set(all_frames)):
        return False

    return True


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
            verbose=verbose-1,
            all_candidate_matches=all_candidate_matches
        )
    return all_neuron_matches, all_confidences, all_candidate_matches


def calc_2frame_matches_using_class(frame0,
                                    frame1,
                                    verbose=1,
                                    use_affine_matching=False,
                                    add_affine_to_candidates=False,
                                    add_gp_to_candidates=False,
                                    DEBUG=False):
    """
    Similar to older function, but this doesn't assume the features are
    already matched

    See also: calc_2frame_matches
    """

    # First, get feature matches
    feature_matches = match_known_features(frame0.all_features,
                                           frame1.all_features,
                                           frame0.keypoints,
                                           frame1.keypoints,
                                           frame0.vol_shape[1:],
                                           frame1.vol_shape[1:],
                                           matches_to_keep=0.5)
    if DEBUG:
        print("All feature matches: ")
        # Draw first 10 matches.
        img1 = frame0.get_data()[15,...]
        img2 = frame1.get_data()[15,...]
        kp1, kp2 = frame0.keypoints, frame1.keypoints
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,feature_matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #plt.figure(figsize=(25,45))
        #plt.imshow(img3),plt.show()
        #[print(f) for f in feature_matches]
        #return img3

    # Second, get neuron matches
    if not use_affine_matching:
        f = calc_matches_using_feature_voting
        feature_matches_dict = extract_map1to2_from_matches(feature_matches)
        opt = {'feature_matches_dict':feature_matches_dict}
    else:
        f = calc_matches_using_affine_propagation
        opt = {'all_feature_matches':feature_matches}
    all_neuron_matches, all_confidences, all_candidate_matches = f(
                                          frame0, frame1,
                                          **opt)

    if add_affine_to_candidates:
        f = calc_matches_using_affine_propagation
        opt = {'all_feature_matches':feature_matches}
        _, _, new_candidate_matches = f(frame0, frame1, **opt)
        all_candidate_matches.extend(new_candidate_matches)

    if add_gp_to_candidates:
        n0 = frame0.neuron_locs.copy()
        n1 = frame1.neuron_locs.copy()

        # TODO: Increase z distances
        n0[:,0] *= 3
        n1[:,0] *= 3
        # Actually match
        opt = {'this_match':all_neuron_matches, 'this_conf':all_confidences}
        matches, _, _ = calc_matches_using_gaussian_process(n0, n1, **opt)
        all_candidate_matches.extend(matches)

    return all_neuron_matches, all_confidences, feature_matches, all_candidate_matches
