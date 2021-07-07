import cv2

from DLC_for_WBFM.utils.external.utils_cv2 import match_object_to_array
from DLC_for_WBFM.utils.feature_detection.class_frame_pair import FramePair
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.feature_detection.utils_features import build_features_1volume, build_feature_tree, \
    build_neuron_tree, build_f2n_map, add_neuron_match, match_known_features, extract_map1to2_from_matches, \
    convert_to_grayscale
from DLC_for_WBFM.utils.feature_detection.utils_affine import calc_matches_using_affine_propagation
from DLC_for_WBFM.utils.feature_detection.utils_detection import detect_neurons_using_ICP, detect_neurons_from_file
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import ReferenceFrame
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings, perform_preprocessing
from DLC_for_WBFM.utils.feature_detection.utils_gaussian_process import calc_matches_using_gaussian_process
from DLC_for_WBFM.utils.feature_detection.utils_networkx import unpack_node_name, is_one_neuron_per_frame
from DLC_for_WBFM.utils.feature_detection.utils_features import keep_top_matches_per_neuron
import numpy as np
import networkx as nx
import collections


##
## Main convinience constructors
##

def build_reference_frame(dat_raw,
                          num_slices,
                          neuron_feature_radius,
                          preprocessing_settings=PreprocessingSettings(),
                          start_slice=2,
                          metadata=None,
                          external_detections=None,
                          verbose=0):
    """Main convenience constructor for ReferenceFrame class"""
    if metadata is None:
        metadata = {}
    dat = perform_preprocessing(dat_raw, preprocessing_settings)

    # Get neurons and features, and a map between them
    neuron_locs = _detect_or_import_neurons(dat, external_detections, metadata, num_slices, start_slice)

    if len(neuron_locs) == 0:
        print("No neurons detected... check data settings")
        raise ValueError
    feature_opt = {'num_features_per_plane': 1000, 'start_plane': 5}
    kps, kp_3d_locs, features = build_features_1volume(dat, **feature_opt)

    # The map requires some open3d subfunctions
    num_f, pc_f, _ = build_feature_tree(kp_3d_locs, which_slice=None)
    _, _, tree_neurons = build_neuron_tree(neuron_locs, to_mirror=False)
    f2n_map = build_f2n_map(kp_3d_locs,
                            num_f,
                            pc_f,
                            neuron_feature_radius,
                            tree_neurons,
                            verbose=verbose - 1)

    # Finally, my summary class
    f = ReferenceFrame(neuron_locs, kps, kp_3d_locs, features, f2n_map,
                       **metadata,
                       preprocessing_settings=preprocessing_settings)
    return f


def _detect_or_import_neurons(dat, external_detections, metadata, num_slices, start_slice):
    if external_detections is None:
        neuron_locs, _, _, _ = detect_neurons_using_ICP(dat,
                                                        num_slices=num_slices,
                                                        alpha=1.0,
                                                        min_detections=3,
                                                        start_slice=start_slice,
                                                        verbose=0)
        neuron_locs = np.array([n for n in neuron_locs])
    else:
        i = metadata['frame_ind']
        neuron_locs = detect_neurons_from_file(external_detections, i)
    return neuron_locs


def build_reference_frame_encoding(dat_raw,
                                   num_slices,
                                   z_depth,
                                   start_slice=None,
                                   metadata={},
                                   external_detections=None,
                                   verbose=0):
    """
    New pipeline that directly builds an embedding for each neuron, instead of detecting keypoints

    See: build_reference_frame
    """
    # DEPRECATE PREPROCESSING
    # dat = perform_preprocessing(dat_raw, preprocessing_settings)
    dat = dat_raw
    neuron_zxy = _detect_or_import_neurons(dat, external_detections, metadata, num_slices, start_slice)

    embeddings, keypoints = encode_all_neurons(neuron_zxy, dat, z_depth)

    # This is now just a trivial mapping
    f2n_map = {i: i for i in range(len(neuron_zxy))}
    f = ReferenceFrame(neuron_zxy, keypoints, neuron_zxy, embeddings, f2n_map,
                       **metadata,
                       preprocessing_settings=None)

    return f


def encode_all_neurons(locs_zxy, im_3d, z_depth):
    """
    Builds a feature vector for each neuron (zxy location) in a 3d volume
    Uses opencv VGG as a 2d encoder for a number of slices above and below the exact z location

    """
    im_3d_gray = [convert_to_grayscale(xy) for xy in im_3d]
    all_embeddings = []
    all_keypoints = []
    encoder = cv2.xfeatures2d.VGG_create()

    # Loop per neuron
    for loc in locs_zxy:
        z, x, y = loc
        kp = cv2.KeyPoint(x, y, 31.0)

        z = int(z)
        all_slices = np.arange(z - z_depth, z + z_depth + 1)
        all_slices = np.clip(all_slices, 0, len(im_3d_gray)-1)
        # Generate features on neighboring z slices as well
        # Repeat slices if near the edge
        ds = []
        for i in all_slices:
            im_2d = im_3d_gray[int(i)]
            _, this_ds = encoder.compute(im_2d, [kp])
            ds.append(this_ds)

        ds = np.hstack(ds)
        all_embeddings.extend(ds)
        all_keypoints.append(kp)

    return np.array(all_embeddings), all_keypoints


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


def calc_2frame_matches_using_class(frame0,
                                    frame1,
                                    verbose=1,
                                    use_affine_matching=False, # DEPRECATED
                                    add_affine_to_candidates=False,
                                    add_gp_to_candidates=False,
                                    DEBUG=False):
    """
    Similar to older function, but this doesn't assume the features are
    already matched

    See also: calc_2frame_matches
    """

    # First, get feature matches
    keypoint_matches = match_known_features(frame0.all_features,
                                            frame1.all_features,
                                            frame0.keypoints,
                                            frame1.keypoints,
                                            frame0.vol_shape[1:],
                                            frame1.vol_shape[1:],
                                            matches_to_keep=1.0,
                                            use_GMS=False)

    # With neuron embeddings, the keypoints are the neurons
    matches_with_conf = match_object_to_array(keypoint_matches, gamma=1.0)

    # if not use_affine_matching:
    #     f = calc_matches_using_feature_voting
    #     keypoint_matches = keep_top_matches_per_neuron(keypoint_matches, frame0, matches_to_keep=0.5)
    #     feature_matches_dict = extract_map1to2_from_matches(keypoint_matches)
    #     opt = {'feature_matches_dict': feature_matches_dict}
    # else:
    #     f = calc_matches_using_affine_propagation
    #     opt = {'all_feature_matches': keypoint_matches}
    # matches_with_conf, all_candidate_matches, _ = f(frame0, frame1, **opt)

    # Create convenience object to store matches
    frame_pair = FramePair(matches_with_conf, matches_with_conf)
    frame_pair.keypoint_matches = matches_with_conf
    # if not use_affine_matching:
    #     frame_pair.feature_matches = all_candidate_matches
    # else:
    #     frame_pair.affine_matches = all_candidate_matches

    # Add additional candidates, if used
    if add_affine_to_candidates:
        opt = {'all_feature_matches': keypoint_matches}
        matches_with_conf, _, affine_pushed = calc_matches_using_affine_propagation(frame0, frame1, **opt)
        frame_pair.affine_matches = matches_with_conf
        frame_pair.affine_pushed_locations = affine_pushed

    if add_gp_to_candidates:
        n0 = frame0.neuron_locs.copy()
        n1 = frame1.neuron_locs.copy()

        # TODO: Increase z distances correctly
        n0[:, 0] *= 3
        n1[:, 0] *= 3
        # Actually match
        opt = {'matches_with_conf': matches_with_conf}
        matches_with_conf, all_gps, gp_pushed = calc_matches_using_gaussian_process(n0, n1, **opt)
        frame_pair.gp_matches = matches_with_conf
        frame_pair.all_gps = all_gps
        frame_pair.gp_pushed_locations = gp_pushed

    return frame_pair
