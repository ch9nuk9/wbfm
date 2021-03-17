from DLC_for_WBFM.utils.feature_detection.utils_features import *
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import *
from DLC_for_WBFM.utils.feature_detection.utils_detection import *
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import *
from DLC_for_WBFM.utils.feature_detection.class_reference_frame import *
from DLC_for_WBFM.utils.feature_detection.utils_candidate_matches import calc_neurons_using_k_cliques, calc_all_bipartite_matches, community_to_matches
from DLC_for_WBFM.utils.feature_detection.utils_networkx import build_digraph_from_matches, unpack_node_name

from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
import copy
import numpy as np
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from collections import defaultdict


##
## Full pipeline
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
    opt = {'num_slices':num_slices,
           'alpha':1.0, # Already multiplied when imported
           'verbose':verbose-1,
           'min_detections':5}
    if neurons0 is None:
        neurons0, _, _, _ = detect_neurons_using_ICP(dat0, **opt)
    if neurons1 is None:
        neurons1, _, _, _ = detect_neurons_using_ICP(dat1, **opt)

    opt = {'verbose':verbose-1,
           'matches_to_keep':0.8,
           'num_features_per_plane':10000,
           'detect_keypoints':True,
           'kp0':neurons0,
           'kp1':neurons1}
    all_f0, all_f1, _, _, _ = build_features_and_match_2volumes(dat0,dat1,**opt)

    # Now, match the neurons using feature space
    opt = {'radius':8,
           'max_nn':50,
           'min_features_needed':5,
           'verbose':verbose-1,
           'to_mirror':False}
    all_matches, _, all_conf = match_centroids_using_tree(np.array(neurons0),
                                                            np.array(neurons1),
                                                            all_f0,
                                                            all_f1,
                                                            **opt)
    return all_matches, all_conf, neurons0, neurons1


def track_neurons_full_video_legacy(vid_fname,
                             start_frame=0,
                             num_frames=10,
                             num_slices=33,
                             alpha=0.15,
                             use_mini_max_projection=False,
                             verbose=0):
    """
    Detects and tracks neurons using opencv-based feature matching
    """
    start_time = time.time()

    # Get initial volume; settings are same for all
    import_opt = {'num_slices':num_slices, 'alpha':alpha}
    dat0 = get_single_volume(vid_fname, start_frame, **import_opt)
    if use_mini_max_projection:
        dat0 = ndi.maximum_filter(dat0, size=(5,1,1))

    # Loop through all pairs
    all_matches = []
    all_conf = []
    all_neurons = []
    previous_neurons = None
    end_frame = start_frame+num_frames
    frame_range = range(start_frame+1, end_frame)
    for i_frame in tqdm(frame_range):
        if verbose >= 1:
            print("===========================================================")
            print(f"Matching frames {i_frame-1} and {i_frame} (end at {end_frame})")
        dat1 = get_single_volume(vid_fname, i_frame, **import_opt)
        if use_mini_max_projection:
            dat1 = ndi.maximum_filter(dat1, size=(5,1,1))

        m, c, n0, n1 = track_neurons_two_volumes(dat0,
                                                  dat1,
                                                  num_slices=num_slices,
                                                  verbose=verbose-1,
                                                  neurons0=previous_neurons)
        all_matches.append(m)
        all_conf.append(c)
        if len(all_neurons)==0:
            # After the first time, n0 doesn't need to be saved
            all_neurons.append(np.array([r for r in n0]))
        all_neurons.append(np.array([r for r in n1]))
        previous_neurons = n1

        dat0 = copy.copy(dat1)
        # dat0 = get_single_volume(vid_fname, i_frame, **import_opt)

    if verbose >= 1:
        total = time.time() - start_time
        print(f"Finished {num_frames} frames in {total} seconds")

    return all_matches, all_conf, all_neurons


##
## Different strategy: reference frames
##

def build_all_reference_frames(num_reference_frames,
                         vid_fname,
                         start_frame,
                         num_frames,
                         num_slices,
                         neuron_feature_radius,
                         start_slice=2,
                         is_sequential=True,
                         preprocessing_settings=PreprocessingSettings(),
                         verbose=1,
                         recalculate_reference_frames=True):
    """
    Selects a sample of reference frames, then builds features for them

    FOR NOW:
    By default, these frames are sequential

    The ref_frames argument allows previously calculated frames to be reused
    """

    other_ind = list(range(start_frame, start_frame+num_frames))
    if is_sequential:
        ref_ind = list(range(start_frame, start_frame+num_reference_frames))
    else:
        ref_ind = random.sample(other_ind, num_reference_frames)
    for ind in ref_ind:
        other_ind.remove(ind)

    ref_dat = []
    ref_frames = []
    video_opt = {'num_slices':num_slices,
                 'alpha':1.0,
                 'dtype':preprocessing_settings.initial_dtype}
    if verbose >= 1:
        print("Building reference frames...")
    for ind in tqdm(ref_ind, total=len(ref_ind)):
        dat = get_single_volume(vid_fname, ind, **video_opt)
        ref_dat.append(dat)

        metadata = {'frame_ind':ind,
                    'vol_shape':dat.shape,
                    'video_fname':vid_fname}
        if recalculate_reference_frames:
            f = build_reference_frame(dat, num_slices, neuron_feature_radius,
                                      start_slice=start_slice,
                                      metadata=metadata,
                                      preprocessing_settings=preprocessing_settings)
            ref_frames.append(f)

    return ref_dat, ref_frames, other_ind


##
## Networkx-based construction of reference indices
##

def neuron_global_id_from_multiple_matches_thresholds(matches, conf, total_frames,
                                           edge_threshs = [0,0.1,0.2,0.3]):
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
        opt = {'reference_ind':reference_ind,'total_frames':total_frames,'thresh':t}
        g2l, l2g, reference_ind, G = add_all_good_components(G,**opt)
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
        k_values = list(range(total_size,2,-1))
        if len(k_values) > 4:
            k_values = k_values[:4]

    # Pre-process matches pairwise by finding best bipartite match
    # TODO: remove hardcoded confidence
    bp_matches = calc_all_bipartite_matches(matches, min_edge_weight=0.2)

    # Get a list of the neuron names that belong to each community
    if verbose >= 1:
        print("Calculating communities. Allowed sizes: ", k_values)
    all_communities = calc_neurons_using_k_cliques(bp_matches,
                                     k_values = k_values,
                                     list_min_sizes = list_min_sizes,
                                     max_size = total_size,
                                     verbose=verbose)

    # Again post-process to enforce unique matches
    #... but I'm pretty sure this just randomly chooses one!
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


def align_dictionaries(ref_set, global2local, local2global):
    """
    Align global and local neuron indices:
        Overwrite keys of 'global2local' and replace with corresponding keys from ref_set

    TODO: for now doesn't allow new neurons to be created
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


def register_all_reference_frames(ref_frames,
                                  previous_ref_set=None,
                                  add_gp_to_candidates=False,
                                  add_affine_to_candidates=False,
                                  use_affine_matching=False,
                                  use_k_cliques=True,
                                  verbose=0):
    """
    Registers a set of reference frames, aligning their neuron indices

    Builds all pairwise matches
        Alternate option: extend a previously built reference_set
    """

    ref_neuron_ind = []
    if previous_ref_set is None:
        pairwise_matches_dict = {}
        feature_matches_dict = {}
        pairwise_conf_dict = {}
        bp_matches_dict = {}
    else:
        previous_ref_set.reference_frames.extend(ref_frames)
        ref_frames = previous_ref_set.reference_frames
        pairwise_matches_dict = previous_ref_set.pairwise_matches
        feature_matches_dict = previous_ref_set.feature_matches
        pairwise_conf_dict = previous_ref_set.pairwise_conf
        bp_matches_dict = previous_ref_set.bipartite_matches

    match_opt = {'use_affine_matching':use_affine_matching,
                 'add_affine_to_candidates':add_affine_to_candidates,
                 'add_gp_to_candidates':add_gp_to_candidates}
    if verbose >= 1:
        print("Pairwise matching all reference frames...")
    for i0, frame0 in tqdm(enumerate(ref_frames), total=len(ref_frames)):
        for i1, frame1 in enumerate(ref_frames):
            key = (i0, i1)
            if i1==i0 and key not in pairwise_matches_dict:
                continue
            out = calc_2frame_matches_using_class(frame0, frame1,**match_opt)
            match, conf, feature_matches, candidate_matches = out
            pairwise_matches_dict[key] = match
            pairwise_conf_dict[key] = conf
            feature_matches_dict[key] = feature_matches
            if candidate_matches is not None:
                bp_matches_dict[key] = list(candidate_matches)
    # Use the matches to build a global index
    if use_k_cliques:
        global2local, local2global = neuron_global_id_from_multiple_matches(
            bp_matches_dict,
            total_size=len(ref_frames),
            verbose=verbose
        )
    else:
        global2local, local2global = neuron_global_id_from_multiple_matches_thresholds(
            pairwise_matches_dict,
            pairwise_conf_dict,
            len(ref_frames)
        )

    # TODO: align to previous global match, if it exists
    if previous_ref_set is not None:
        global2local, local2global = align_dictionaries(
            previous_ref_set,
            global2local,
            local2global
        )

    # Update the global indices of the individual reference frames
    for f in ref_frames:
        f.neuron_ids = []
    for local_ind, global_ind in local2global.items():
        frame_ind, local_neuron = local_ind
        ref_frames[frame_ind].neuron_ids.append([local_neuron, global_ind])

    # Build a class to store all the information
    reference_set = RegisteredReferenceFrames(
        global2local,
        local2global,
        ref_frames,
        pairwise_matches_dict,
        pairwise_conf_dict,
        feature_matches_dict,
        bp_matches_dict
    )

    return reference_set
    #return global2local, local2global, pairwise_matches_dict, pairwise_conf_dict, feature_matches_dict, bp_matches_dict


def match_to_reference_frames(this_frame, reference_set):
    """
    Registers a single frame to a set of references
    """

    # Build a map from this frame's indices to the global neuron frame
    all_global_matches = []
    all_conf = []
    for ref in reference_set.reference_frames:
        # Get matches (coordinates are local to this reference frame)
        # TODO: only attempt to check the subset of reference neurons
        local_matches, conf, _, _ = calc_2frame_matches_using_class(this_frame, ref)
        # Convert to global coordinates
        global_matches = []
        frame_ind = ref.frame_ind
        l2g = reference_set.local2global
        for m in local_matches:
            ref_neuron_ind = m[1]
            global_ind = l2g.get((frame_ind, ref_neuron_ind), None)
            # The matched neuron may not be part of the actual reference set
            if global_ind is not None:
                global_matches.append([m[0], global_ind])
        all_global_matches.append(global_matches)
        all_conf.append(conf)

    # Compact the matches of each reference frame ID into a single list
    per_neuron_matches = defaultdict(list)
    for frame_match in all_global_matches:
        for neuron_matches in frame_match:
            per_neuron_matches[neuron_matches[0]].append(neuron_matches[1])

    # Then, use the matches to vote for the best neuron
    # TODO: use graph connected components
    final_matches = []
    final_conf = []
    min_features_needed = len(reference_set.reference_frames)/2.0
    for this_local_ind, these_matches in per_neuron_matches.items():
        final_matches, final_conf, _ = add_neuron_match(
            final_matches,
            final_conf,
            this_local_ind,
            min_features_needed,
            these_matches
        )

    return final_matches, final_conf, per_neuron_matches


def match_all_to_reference_frames(reference_set,
                                  vid_fname,
                                  other_ind,
                                  video_opt,
                                  metadata,
                                  num_slices,
                                  neuron_feature_radius,
                                  preprocessing_settings):
    """
    Multi-frame wrapper around match_to_reference_frames()
    """

    all_matches = []
    all_other_frames = []

    for ind in tqdm(other_ind, total=len(other_ind)):
        dat = get_single_volume(vid_fname, ind, **video_opt)
        metadata['frame_ind'] = ind

        f = build_reference_frame(dat, num_slices, neuron_feature_radius,
                              metadata=metadata,
                              preprocessing_settings=preprocessing_settings)
        matches, _, per_neuron_matches = match_to_reference_frames(f, reference_set)

        all_matches.append(matches)
        f.neuron_ids = per_neuron_matches
        all_other_frames.append(f)

    return all_matches, all_other_frames


##
## Full pipeline function
##


def track_neurons_full_video(vid_fname,
                             start_frame=0,
                             num_frames=10,
                             num_slices=33,
                             neuron_feature_radius=5.0,
                             preprocessing_settings=PreprocessingSettings(),
                             use_affine_matching=False,
                             add_affine_to_candidates=False,
                             add_gp_to_candidates=False,
                             save_candidate_matches=False,
                             external_detections=None,
                             verbose=0):
    """
    Detects and tracks neurons using opencv-based feature matching
    Note: only compares adjacent frames
        Thus, if a neuron is lost in a single frame, the track ends

    New: uses and returns my class of features
    """
    # Get initial volume; settings are same for all
    import_opt = {'num_slices':num_slices,
                 'alpha':1.0,
                 'dtype':preprocessing_settings.initial_dtype}
    ref_opt = {'neuron_feature_radius':neuron_feature_radius}
    def local_build_frame(frame_ind,
                          vid_fname=vid_fname,
                          import_opt=import_opt,
                          ref_opt=ref_opt,
                          external_detections=external_detections):
        dat = get_single_volume(vid_fname, frame_ind, **import_opt)
        metadata = {'frame_ind':frame_ind,
                    'vol_shape':dat.shape,
                    'video_fname':vid_fname}
        f = build_reference_frame(dat,
                                  num_slices=import_opt['num_slices'],
                                  **ref_opt,
                                  metadata=metadata,
                                  preprocessing_settings=preprocessing_settings,
                                  external_detections=external_detections)
        return f

    if verbose >= 1:
        print("Building initial frame...")
    frame0 = local_build_frame(start_frame)

    # Loop through all pairs
    pairwise_matches_dict = {}
    pairwise_candidates_dict = {}
    pairwise_conf_dict = {}
    # all_frames = [frame0]
    all_frame_dict = {start_frame:frame0}
    end_frame = start_frame+num_frames
    frame_range = range(start_frame+1, end_frame)
    match_opt = {'use_affine_matching':use_affine_matching,
                 'add_affine_to_candidates':add_affine_to_candidates,
                 'add_gp_to_candidates':add_gp_to_candidates}
    for i_frame in tqdm(frame_range):
        frame1 = local_build_frame(i_frame)

        out = calc_2frame_matches_using_class(frame0, frame1,**match_opt)
        match, conf, fm, candidates = out
        # Save to dictionaries
        key = (i_frame-1, i_frame)
        pairwise_matches_dict[key] = match
        pairwise_conf_dict[key] = conf
        if save_candidate_matches:
            pairwise_candidates_dict[key] = candidates
        # Save frame to list
        # all_frames.append(frame1)
        all_frame_dict[i_frame] = frame1
        frame0 = frame1

    return pairwise_matches_dict, pairwise_conf_dict, all_frame_dict, pairwise_candidates_dict


def track_via_reference_frames(vid_fname,
                               start_frame=0,
                               num_frames=10,
                               num_slices=33,
                               neuron_feature_radius=5.0,
                               start_slice=2,
                               verbose=0,
                               num_reference_frames=5,
                               add_gp_to_candidates=False,
                               add_affine_to_candidates=False,
                               use_affine_matching=False,
                               use_k_cliques=True,
                               preprocessing_settings=PreprocessingSettings(),
                               reference_set=None):
    """
    Tracks neurons by registering them to a set of reference frames
    """

    # First, analyze the reference frames
    if verbose >= 1:
        print("Loading reference frames...")
    video_opt = {'vid_fname':vid_fname,
                 'start_frame':start_frame,
                 'num_frames':num_frames,
                 'num_slices':num_slices,
                 'neuron_feature_radius':neuron_feature_radius,
                 'verbose':verbose-1}
    if reference_set is None:
        ref_dat, ref_frames, other_ind = build_all_reference_frames(
            num_reference_frames,
            **video_opt,
            preprocessing_settings=preprocessing_settings,
            recalculate_reference_frames=True
        )
    else:
        # Reuse previous reference frames, but still build the metadata
        ref_dat, _, other_ind = build_all_reference_frames(
            num_reference_frames,
            **video_opt,
            preprocessing_settings=preprocessing_settings,
            recalculate_reference_frames=False
        )
        ref_frames = reference_set.reference_frames

    if verbose >= 1:
        print("Analyzing reference frames...")
    match_opt = {'use_affine_matching':use_affine_matching,
                 'add_affine_to_candidates':add_affine_to_candidates,
                 'add_gp_to_candidates':add_gp_to_candidates,
                 'use_k_cliques':use_k_cliques,
                 'verbose':verbose-1}
    if reference_set is None:
        reference_set = register_all_reference_frames(ref_frames, **match_opt)

    if verbose >= 1:
        print("Matching other frames to reference...")
    video_opt = {'num_slices':num_slices,
                 'alpha':1.0,
                 'dtype':preprocessing_settings.initial_dtype}
    metadata = ref_frames[0].get_metadata()
    all_matches, all_other_frames = match_all_to_reference_frames(
        reference_set,
        vid_fname,
        other_ind,
        video_opt,
        metadata,
        num_slices,
        neuron_feature_radius,
        preprocessing_settings=preprocessing_settings
    )

    return all_matches, all_other_frames, reference_set



def track_neurons_full_video_window(vid_fname,
                             start_frame=0,
                             num_frames=10,
                             num_slices=33,
                             neuron_feature_radius=5.0,
                             preprocessing_settings=PreprocessingSettings(),
                             num_subsequent_matches=2,
                             use_affine_matching=False,
                             add_affine_to_candidates=False,
                             add_gp_to_candidates=False,
                             save_candidate_matches=False,
                             external_detections=None,
                             verbose=0):
    """
    Detects and tracks neurons using opencv-based feature matching
    Compares each frame to the next (num_subsequent_matches) frames

    See also: track_neurons_full_video
    """
    # Get initial volume; settings are same for all
    import_opt = {'num_slices':num_slices,
                 'alpha':1.0,
                 'dtype':preprocessing_settings.initial_dtype}
    ref_opt = {'neuron_feature_radius':neuron_feature_radius}
    def local_build_frame(frame_ind,
                          vid_fname=vid_fname,
                          import_opt=import_opt,
                          ref_opt=ref_opt,
                          external_detections=external_detections):
        dat = get_single_volume(vid_fname, frame_ind, **import_opt)
        metadata = {'frame_ind':frame_ind,
                    'vol_shape':dat.shape,
                    'video_fname':vid_fname}
        f = build_reference_frame(dat,
                                  num_slices=import_opt['num_slices'],
                                  **ref_opt,
                                  metadata=metadata,
                                  preprocessing_settings=preprocessing_settings,
                                  external_detections=external_detections)
        return f

    if verbose >= 1:
        print("Building initial frame...")

    # Loop through all pairs
    pairwise_matches_dict = {}
    pairwise_candidates_dict = {}
    pairwise_conf_dict = {}
    all_frame_dict = {}
    end_frame = start_frame+num_frames
    frame_range = list(range(start_frame, end_frame))
    match_opt = {'use_affine_matching':use_affine_matching,
                 'add_affine_to_candidates':add_affine_to_candidates,
                 'add_gp_to_candidates':add_gp_to_candidates}
    for i_base_frame in tqdm(frame_range):
        # Check if we already built the frame
        if i_base_frame in all_frame_dict:
            base_frame = all_frame_dict[i_base_frame]
        else:
            base_frame = local_build_frame(i_base_frame)
            all_frame_dict[i_base_frame] = base_frame
        window_range = range(i_base_frame+1, i_base_frame+num_subsequent_matches+1)
        for i_next_frame in window_range:
            if i_next_frame in all_frame_dict:
                next_frame = all_frame_dict[i_next_frame]
            else:
                next_frame = local_build_frame(i_next_frame)
                all_frame_dict[i_next_frame] = next_frame

            out = calc_2frame_matches_using_class(base_frame, next_frame,**match_opt)
            match, conf, fm, candidates = out
            # Save to dictionaries
            key = (i_base_frame, i_next_frame)
            pairwise_matches_dict[key] = match
            pairwise_conf_dict[key] = conf
            if save_candidate_matches:
                pairwise_candidates_dict[key] = candidates

    return pairwise_matches_dict, pairwise_conf_dict, all_frame_dict, pairwise_candidates_dict










def track_via_sequence_consensus(vid_fname,
                                 start_frame=0,
                                 num_frames=10,
                                 num_slices=33,
                                 neuron_feature_radius=5.0,
                                 verbose=0,
                                 num_consensus_frames=3,
                                 preprocessing_settings=PreprocessingSettings()):
    """
    OLD

    Tracks neurons by finding consensus between a sliding window of frames

    Note: if num_consensus_frames=2, this is tracking via adjacent frames
    """

    # Initial frame calculations

    # Build a reference set of the first n-1 frames
    video_opt = {'vid_fname':vid_fname,
                 'start_frame':start_frame,
                 'num_frames':num_frames,
                 'num_slices':num_slices,
                 'neuron_feature_radius':neuron_feature_radius,
                 'verbose':verbose-1}
    _, ref_frames, _ = build_all_reference_frames(
        num_consensus_frames-1,
        **video_opt,
        preprocessing_settings=preprocessing_settings
    )
    reference_set_minus1 = register_all_reference_frames(ref_frames)

    all_frames = reference_set_minus1.reference_frames.copy()
    ind = range(start_frame+num_consensus_frames, start_frame+num_frames)
    frame_video_opt = {'num_slices':num_slices,
                  'alpha':1.0,
                  'dtype':preprocessing_settings.initial_dtype}
    metadata = ref_frames[0].get_metadata()
    for i_frame in tqdm(ind, total=len(ind)):
        # Build the next frame
        metadata['frame_ind'] = i_frame
        dat = get_single_volume(vid_fname, i_frame, **frame_video_opt)
        next_frame = build_reference_frame(dat, num_slices, neuron_feature_radius,
                                           metadata=metadata,
                                           preprocessing_settings=preprocessing_settings)
        # Match this frame
        reference_set = register_all_reference_frames(
            [next_frame],
            previous_ref_set=reference_set_minus1
        )

        # Adjust by 1: the new reference set partially overlaps with the previous
        reference_set_minus1 = remove_first_frame(reference_set)

        all_frames.append(reference_set.reference_frames[-1])

    return all_frames


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
    all_starts = clust_df['slice_ind'].apply(lambda x : x[0])
    all_ends = clust_df['slice_ind'].apply(lambda x : x[-1])
    all_long_enough = np.where(all_ends - all_starts > min_starting_tracklet_length)[0]
    is_available = clust_df['slice_ind'].apply(lambda x : True)

    # Reuse distant matches calculations
    tracklet_matches = []
    tracklet_conf = []

    for ind in tqdm(all_long_enough):
        # Get frame and individual neuron to match
        i_end_frame = all_ends.at[ind]
        frame0 = all_frames[i_end_frame]
        neuron0 = clust_df.at[ind,'all_ind_local'][-1]

        # Get all close-by starts
        start_is_after = all_starts.gt(i_end_frame+1)
        start_is_close = all_starts.lt(max_stitch_distance+i_end_frame)
        tmp = start_is_after & start_is_close & is_available
        possible_start_tracks = np.where(tmp)[0]
        if len(possible_start_tracks)==0:
            continue

        # Loop through possible next tracklets
        for i_start_track in possible_start_tracks:
            i_start_frame = all_starts.at[i_start_track]
            frame1 = all_frames[i_start_frame]
            neuron1 = clust_df.at[i_start_track,'all_ind_local'][0]

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
                out = calc_2frame_matches_using_class(frame0, frame1)
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
                    #2err
                    continue
                tracklet_matches.append(t_key)
                tracklet_conf.append(this_conf)
                is_available.at[i_start_track] = False
                # TODO: just take the first match
                break

    df = consolidate_tracklets(clust_df.copy(), tracklet_matches, verbose)
    if verbose >= 1:
        print("Finished")
    intermediates = (distant_matches_dict, distant_conf_dict, tracklet_matches, all_starts, all_ends)
    return df, intermediates
