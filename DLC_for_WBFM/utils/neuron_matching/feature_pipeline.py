import logging
from collections import defaultdict
from typing import Tuple, Dict
import concurrent.futures
import numpy as np
import zarr as zarr
from tqdm import tqdm

from DLC_for_WBFM.utils.neuron_matching.class_frame_pair import FramePair, calc_FramePair_from_Frames, \
    FramePairOptions
from DLC_for_WBFM.utils.neuron_matching.class_reference_frame import RegisteredReferenceFrames, ReferenceFrame, \
    build_reference_frame_encoding
from DLC_for_WBFM.utils.general.custom_errors import NoMatchesError, NoNeuronsError
from DLC_for_WBFM.utils.neuron_matching.utils_candidate_matches import calc_neurons_using_k_cliques, \
    calc_all_bipartite_matches, calc_neuron_using_voronoi
from DLC_for_WBFM.utils.neuron_matching.utils_detection import detect_neurons_using_ICP
from DLC_for_WBFM.utils.neuron_matching.utils_features import build_features_and_match_2volumes, \
    match_centroids_using_tree
from DLC_for_WBFM.utils.external.utils_networkx import build_digraph_from_matches, unpack_node_name
from DLC_for_WBFM.utils.neuron_matching.utils_reference_frames import add_all_good_components, \
    is_ordered_subset
from DLC_for_WBFM.utils.tracklets.utils_tracklets import consolidate_tracklets
from DLC_for_WBFM.utils.general.preprocessing.utils_tif import PreprocessingSettings

from segmentation.util.utils_metadata import DetectedNeurons

##
## Full traces
##


def track_neurons_two_volumes(dat0,
                              dat1,
                              num_slices=33,
                              neurons0=None,
                              neurons1=None,
                              external_detections=None,
                              verbose=1):
    """
    Matches neurons between two volumes

    Can use previously detected neurons, if passed
    """
    # Detect neurons, then features for each volume
    options = {'num_slices': num_slices,
               'alpha': 1.0,  # Already multiplied when imported
               'verbose': verbose - 1,
               'min_detections': 5}
    if neurons0 is None:
        neurons0, _, _, _ = detect_neurons_using_ICP(dat0, **options)
    if neurons1 is None:
        neurons1, _, _, _ = detect_neurons_using_ICP(dat1, **options)

    options = {'verbose': verbose - 1,
               'matches_to_keep': 0.2,
               'num_features_per_plane': 10000,
               'detect_keypoints': True,
               'kp0': neurons0,
               'kp1': neurons1}
    all_f0, all_f1, _, _, _, _ = build_features_and_match_2volumes(dat0, dat1, **options)

    # Now, match the neurons using feature space
    options = {'radius': 8,
               'max_nn': 50,
               'min_features_needed': 5,
               'verbose': verbose - 1,
               'to_mirror': False}
    all_matches, _, all_conf = match_centroids_using_tree(np.array(neurons0),
                                                          np.array(neurons1),
                                                          all_f0,
                                                          all_f1,
                                                          **options)
    return all_matches, all_conf, neurons0, neurons1

##
## Networkx-based construction of reference indices
##

def neuron_global_id_from_multiple_matches_thresholds(matches, conf, total_frames,
                                                      edge_threshs=[0, 0.1, 0.2, 0.3]):
    """
    Builds a vector of neuron matches from pairwise matchings to multiple frames
        Uses a list of thresholds to check for strongly connected components
        DEPRECATED

    Input format:
        matches[T] -> match_array
        where T is a tuple indexing the pairwise matches, e.g. (1,2)
        and match_array is an 'n x 2' array of the neuron indices in that frame

    Algorithm:
        Builds a directed graph from all individual frame's neurons
        Neurons with the same identity are connected components
        If this doesn't fully separate them, then the remaining ones are clustered

    Output format:
        global_ind_dict[frame_ind] -> neuron_vector_dict
        where 'frame_ind' is the frame whose indices are
    """

    G = build_digraph_from_matches(matches, conf, verbose=0)
    global2local = {}
    local2global = {}
    reference_ind = 0

    for t in edge_threshs:
        options = {'reference_ind': reference_ind, 'total_frames': total_frames, 'thresh': t}
        g2l, l2g, reference_ind, G = add_all_good_components(G, **options)
        global2local.update(g2l)
        local2global.update(l2g)

    return global2local, local2global


def neuron_global_id_from_multiple_matches(matches,
                                           total_size=None,
                                           k_values=None,
                                           list_min_sizes=None,
                                           verbose=1):
    """
    Builds a consistent set of neuron IDs using k-clique clustering
    """
    if list_min_sizes is None and k_values is None:
        list_min_sizes = [2]
        k_values = list(range(total_size, 2, -1))
        if len(k_values) > 4:
            k_values = k_values[:4]

    # Pre-process matches pairwise by finding best bipartite match
    # OPTIMIZE: remove hardcoded confidence
    bp_matches = calc_all_bipartite_matches(matches, min_edge_weight=0.2)

    # Get a list of the neuron names that belong to each community
    if verbose >= 1:
        print("Calculating communities. Allowed sizes: ", k_values)
    all_communities = calc_neurons_using_k_cliques(bp_matches,
                                                   k_values=k_values,
                                                   list_min_sizes=list_min_sizes,
                                                   max_size=total_size,
                                                   verbose=verbose)

    # Again post-process to enforce unique matches
    # ... but I'm pretty sure this just randomly chooses one!
    # clique_matches = community_to_matches(all_communities)
    # clique_matches = calc_all_bipartite_matches(clique_matches)

    # Build output format
    local2global = {}
    global2local = defaultdict(list)
    for i_comm, this_comm in enumerate(all_communities):
        for this_neuron in this_comm:
            key = unpack_node_name(this_neuron)
            # Map from local to community index = global neuron ID
            local2global[key] = i_comm
            # Opposite direction: from global ID to list of local indices
            frame_ind, neuron_ind = key
            global2local[i_comm].append(neuron_ind)

    return global2local, local2global


def neuron_global_id_from_multiple_matches_voronoi(matches, conf, total_frames,
                                                   verbose=0):
    """
    Builds global ID based on voronoi cells
    """

    # Convert confidences to distance
    dist = {}
    for k, v in conf.items():
        these_dist = [(1.0 / conf) for conf in v]
        dist[k] = these_dist
    # Actual clustering
    global2local = calc_neuron_using_voronoi(matches,
                                             dist,
                                             total_frames,
                                             target_size_vec=None,
                                             verbose=verbose)
    # Align formatting
    local2global = {}
    for global_ind, v in global2local.items():
        for node in v:
            key = unpack_node_name(node)
            local2global[key] = global_ind

    return global2local, local2global


def align_dictionaries(ref_set, global2local, local2global):
    """
    Align global and local neuron indices:
        Overwrite keys of 'global2local' and replace with corresponding keys from ref_set

    WARNING: for now doesn't allow new neurons to be created
    """

    g2l_out = {}
    l2g_out = {}

    old2new = {}
    for key_true, val_true in ref_set.global2local.items():
        # Check which this corresponds to in the new dict
        for key_tmp, val_tmp in global2local.items():
            if is_ordered_subset(val_true, val_tmp):
                old2new[key_tmp] = key_true
                g2l_out[key_true] = val_tmp
                break
    # Same for other dict
    for key, val in local2global.items():
        if val in old2new:
            l2g_out[key] = old2new[val]

    return g2l_out, l2g_out


##
## Matching the features of the frames
##

def create_dict_from_matches(self):
    assert type(self) == RegisteredReferenceFrames

    all_cluster_modes = ['k_clique', 'threshold', 'voronoi']
    if self.neuron_cluster_mode == 'k_clique':
        global2local, local2global = neuron_global_id_from_multiple_matches(
            self.bipartite_matches,
            total_size=len(self.reference_frames),
            verbose=self.verbose
        )
    elif self.neuron_cluster_mode == 'threshold':
        global2local, local2global = neuron_global_id_from_multiple_matches_thresholds(
            self.pairwise_matches,
            self.pairwise_conf,
            len(self.reference_frames)
        )
    elif self.neuron_cluster_mode == 'voronoi':
        global2local, local2global = neuron_global_id_from_multiple_matches_voronoi(
            self.pairwise_matches,
            self.pairwise_conf,
            len(self.reference_frames),
            verbose=self.verbose
        )
    else:
        print("Unrecognized cluster mode; finishing without global neuron labels")
        print(f"Allowed cluster modes are: {all_cluster_modes}")

    self.global2local = global2local
    self.local2global = local2global


# def match_to_reference_frames(this_frame, reference_set, min_conf=1.0):
#     """
#     Registers a single frame to a set of references
#     """
#
#     # Build a map from this frame's indices to the global neuron frame
#     all_global_matches = []
#     all_conf = []
#     for ref_frame_ind, ref in reference_set.reference_frames.items():
#         # Get matches (coordinates are local to this reference frame)
#         # OPTMIZE: only attempt to check the subset of reference neurons
#         local_matches, conf, _, _ = calc_FramePair_from_Frames(this_frame, ref, None)
#         # Convert to global coordinates
#         global_matches = []
#         global_conf = []
#         l2g = reference_set.local2global
#         for m, c in zip(local_matches, conf):
#             # Check each match between the test frame and the current ref
#             ref_neuron_ind = m[1]
#             global_ind = l2g.get((ref_frame_ind, ref_neuron_ind), None)
#             # The matched neuron may not be part of the actual reference set
#             if global_ind is not None and c > min_conf:
#                 global_matches.append([m[0], global_ind])
#                 global_conf.append(c)
#         all_global_matches.append(global_matches)
#         all_conf.append(conf)
#
#     # Different approach: bipartite matching between reference set and each frame
#     edges_dict = defaultdict(int)
#     for frame_match, frame_conf in zip(all_global_matches, all_conf):
#         for neuron_matches, neuron_conf in zip(frame_match, frame_conf):
#             key = (neuron_matches[0], neuron_matches[1])
#             # COMBAK: add conf
#             edges_dict[key] += neuron_conf
#     edges = [[k[0], k[1], v] for k, v in edges_dict.items()]
#     all_bp_matches = calc_bipartite_matches(edges)
#
#     # TODO: fix last return value
#     return all_bp_matches, all_conf, edges

##
## Full traces function
##


def track_neurons_full_video(video_fname: str, start_volume: int = 0, num_frames: int = 10,
                             z_depth_neuron_encoding: float = 5.0,
                             preprocessing_settings: PreprocessingSettings = PreprocessingSettings(),
                             pairwise_matches_params: FramePairOptions = None,
                             external_detections: str = None, verbose: int = 0) -> Tuple[Dict[Tuple[int, int], FramePair], Dict[int, ReferenceFrame]]:
    """
    Detects and tracks neurons using opencv-based feature matching
    Note: only compares adjacent frames
        Thus, if a neuron is lost in a single frame, the track ends

    New: uses and returns my class of features
    """
    if preprocessing_settings is not None:
        dtype = preprocessing_settings.initial_dtype
        raise DeprecationWarning("Preprocessing on individual frames is deprecated")
    else:
        # TODO: better way to get datatype
        dtype = 'uint8'

    # Build frames, then match them
    end_volume = start_volume + num_frames
    all_frame_dict = calculate_frame_objects_full_video(external_detections, start_volume, end_volume,
                                                        video_fname, z_depth_neuron_encoding)

    try:
        all_frame_pairs = match_all_adjacent_frames(all_frame_dict, end_volume, pairwise_matches_params, start_volume)
        return all_frame_pairs, all_frame_dict
    except (ValueError, NoNeuronsError, NoMatchesError) as e:
        logging.warning("Error in frame pair matching; quitting gracefully and saving the frame pairs:")
        print(e)
        return None, all_frame_dict


def match_all_adjacent_frames(all_frame_dict, end_volume, pairwise_matches_params, start_volume):
    all_frame_pairs = {}
    frame_range = range(start_volume + 1, end_volume)
    logging.info(f"Calculating Frame pairs for frames:  {start_volume + 1} to {end_volume}")
    for i_frame in tqdm(frame_range):
        key = (i_frame - 1, i_frame)
        frame0, frame1 = all_frame_dict[key[0]], all_frame_dict[key[1]]
        this_pair = calc_FramePair_from_Frames(frame0, frame1, frame_pair_options=pairwise_matches_params)

        all_frame_pairs[key] = this_pair
    return all_frame_pairs


def calculate_frame_objects_full_video(external_detections, start_volume, end_volume, video_fname,
                                       z_depth_neuron_encoding, encoder_opt=None, max_workers=8, **kwargs):
    # Get initial volume; settings are same for all
    vid_dat = zarr.open(video_fname, synchronizer=zarr.ThreadSynchronizer())
    vol_shape = vid_dat[0, ...].shape
    all_detected_neurons = DetectedNeurons(external_detections)
    all_detected_neurons.setup()

    def _build_frame(frame_ind: int) -> ReferenceFrame:
        metadata = {'frame_ind': frame_ind,
                    'vol_shape': vol_shape,
                    'video_fname': video_fname,
                    'z_depth': z_depth_neuron_encoding}
        f = build_reference_frame_encoding(metadata=metadata, all_detected_neurons=all_detected_neurons,
                                           encoder_opt=encoder_opt)
        return f

    # Build all frames initially, then match
    frame_range = range(start_volume, end_volume)
    all_frame_dict = dict()
    logging.info(f"Calculating Frame objects for frames: {start_volume} to {end_volume}")
    with tqdm(total=len(frame_range)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_build_frame, i): i for i in frame_range}
            for future in concurrent.futures.as_completed(futures):
                i_frame = futures[future]
                all_frame_dict[i_frame] = future.result()
                pbar.update(1)
    return all_frame_dict


##
## Postprocessing
##

def stitch_tracklets(clust_df,
                     all_frames,
                     max_stitch_distance=10,
                     min_starting_tracklet_length=3,
                     minimum_match_confidence=0.4,
                     distant_matches_dict={},
                     distant_conf_dict={},
                     verbose=0):
    """
    Takes tracklets in a dataframe and attempts to stitch them together
    Uses list of original frame data

    Only attempts to match the last frame to first frames of other tracklets

    Can pass distant_matches_dict and distant_conf_dict from a previous run
    """
    if verbose >= 1:
        print(f"Trying to consolidate {clust_df.shape[0]} tracklets")
        print("Note: computational time of this function is front-loaded")
    # Get tracklet starting and ending indices in frame space
    all_starts = clust_df['slice_ind'].apply(lambda x: x[0])
    all_ends = clust_df['slice_ind'].apply(lambda x: x[-1])
    all_long_enough = np.where(all_ends - all_starts > min_starting_tracklet_length)[0]
    is_available = clust_df['slice_ind'].apply(lambda x: True)

    # Reuse distant matches calculations
    tracklet_matches = []
    tracklet_conf = []

    for ind in tqdm(all_long_enough):
        # Get frame and individual neuron to match
        i_end_frame = all_ends.at[ind]
        frame0 = all_frames[i_end_frame]
        neuron0 = clust_df.at[ind, 'all_ind_local'][-1]

        # Get all close-by starts
        start_is_after = all_starts.gt(i_end_frame + 1)
        start_is_close = all_starts.lt(max_stitch_distance + i_end_frame)
        tmp = start_is_after & start_is_close & is_available
        possible_start_tracks = np.where(tmp)[0]
        if len(possible_start_tracks) == 0:
            continue

        # Loop through possible next tracklets
        for i_start_track in possible_start_tracks:
            i_start_frame = all_starts.at[i_start_track]
            frame1 = all_frames[i_start_frame]
            neuron1 = clust_df.at[i_start_track, 'all_ind_local'][0]

            if verbose >= 4:
                print(f"Trying to match tracklets {ind} and {i_start_track}")
            key = (i_end_frame, i_start_frame)
            if key in distant_matches_dict:
                matches = distant_matches_dict[key]
                conf = distant_conf_dict[key]
            else:
                # Otherwise, calculate from scratch
                if verbose >= 3:
                    print(f"Calculating new matches between frames {key}")
                out = calc_FramePair_from_Frames(frame0, frame1, None)
                raise ValueError("Needs refactor with FramePair")
                matches, conf = out[0], out[1]
                # Save for future
                distant_matches_dict[key] = matches
                distant_conf_dict[key] = conf

            # Find if these specific neurons are matched in the frames
            n_key = [neuron0, neuron1]
            t_key = [ind, i_start_track]
            if n_key in matches:
                this_conf = conf[matches.index(n_key)]
                if verbose >= 2:
                    print(f"Matched tracks {t_key} with confidence {this_conf}")
                    print(f"(frames {key} and neurons {n_key})")
                if this_conf < minimum_match_confidence:
                    # 2err
                    continue
                tracklet_matches.append(t_key)
                tracklet_conf.append(this_conf)
                is_available.at[i_start_track] = False
                # OPTIMIZE: just take the first match
                break

    df = consolidate_tracklets(clust_df.copy(), tracklet_matches, verbose)
    if verbose >= 1:
        print("Finished")
    intermediates = (distant_matches_dict, distant_conf_dict, tracklet_matches, all_starts, all_ends)
    return df, intermediates
