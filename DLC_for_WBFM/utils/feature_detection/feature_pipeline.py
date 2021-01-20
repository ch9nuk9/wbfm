from DLC_for_WBFM.utils.feature_detection.utils_features import *
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import *
from DLC_for_WBFM.utils.feature_detection.utils_detection import *
from DLC_for_WBFM.utils.feature_detection.utils_reference_frames import *

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
                              verbose=1):
    """
    Matches neurons between two volumes

    Can use previously detected neurons, if passed
    """
    # Detect neurons, then features for each volume
    opt = {'num_slices':num_slices,
           'alpha':1.0, # Already multiplied when imported
           'verbose':verbose-1}
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
    all_f0, all_f1, _, _ = build_features_and_match_2volumes(dat0,dat1,**opt)

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


def track_neurons_full_video(vid_fname,
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
    for i_frame in frame_range:
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
                         alpha,
                         is_sequential=True,
                         do_mini_max_projections=True,
                         verbose=1):
    """
    Selects a sample of reference frames, then builds features for them

    FOR NOW:
    By default, these frames are sequential
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
                 'alpha':alpha}
    if verbose >= 1:
        print("Building reference frames...")
    for ind in tqdm(ref_ind, total=len(ref_ind)):
        dat = get_single_volume(vid_fname, ind, **video_opt)
        ref_dat.append(dat)

        metadata = {'frame_ind':ind,
                    'vol_shape':dat.shape,
                    'video_fname':vid_fname,
                    'alpha':alpha}
        f = build_reference_frame(dat, num_slices, neuron_feature_radius,
                                  metadata=metadata)
        ref_frames.append(f)

    return ref_dat, ref_frames, other_ind


def calc_2frame_matches_using_class(frame0,
                                    frame1,
                                    use_bipartite_matching=False,
                                    verbose=1,
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
    feature_matches_dict = extract_map1to2_from_matches(feature_matches)
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
    all_neuron_matches = []
    all_confidences = []
    all_candidate_matches = []
    for neuron0_ind, neuron0_loc in enumerate(frame0.iter_neurons()):
        # Get features of this neuron
        this_f0 = frame0.get_features_of_neuron(neuron0_ind)
        if DEBUG:
            print(f"=======Neuron {neuron0_ind}=========")
            #print("Features in vol0: ", this_f0)
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

        min_features_needed = 2 # TODO
        all_neuron_matches, all_confidences, all_candidate_matches = add_neuron_match(
            all_neuron_matches,
            all_confidences,
            neuron0_ind,
            min_features_needed,
            this_n1,
            verbose=verbose-1,
            all_candidate_matches=all_candidate_matches
        )
        if DEBUG:
            break

    if use_bipartite_matching:
        all_bp_matches = calc_bipartite_matches(all_candidate_matches, verbose-1)
    else:
        all_bp_matches = None

    return all_neuron_matches, all_confidences, feature_matches, all_bp_matches


##
## Networkx-based construction of reference indices
##

def neuron_global_id_from_multiple_matches(matches, conf, total_frames,
                                           edge_threshs = [0,0.1,0.2,0.3]):
    """
    Builds a vector of neuron matches from pairwise matchings to multiple frames

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


def align_dictionaries(ref_set, global2local, local2global):
    """
    Align global and local neuron indices:
        Overwrite keys of 'global2local' and replace with corresponding keys from ref_set

    TODO: for now doesn't allow new neurons to be created
    """

    g2l_out = {}
    l2g_out = {}

    old2new = {}
    for key_true, val_true in ref_set.global2local:
        # Check which this corresponds to in the new dict
        for key_tmp, val_tmp in global2local:
            if is_ordered_subset(val_true, val_tmp):
                old2new[key_tmp] = key_true
                g2l_out[key_true] = val_tmp
                break
    # Same for other dict
    for key_tmp, val in local2global:
        key = old2new[key_tmp]
        l2g_out[key] = val

    return g2l_out, l2g_out

##
## Matching the features of the frames
##


def register_all_reference_frames(ref_frames,
                                  use_bipartite_matching=False,
                                  previous_ref_set=None,
                                  verbose=1):
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
        pairwise_matches_dict = previous_ref_set.pairwise_matches
        feature_matches_dict = previous_ref_set.feature_matches
        pairwise_conf_dict = previous_ref_set.pairwise_conf
        bp_matches_dict = previous_ref_set.bp_matches

    if verbose >= 1:
        print("Pairwise matching all reference frames...")
    for i0, frame0 in tqdm(enumerate(ref_frames), total=len(ref_frames)):
        for i1, frame1 in enumerate(ref_frames):
            key = (i0, i1)
            if i1==i0 and key not in pairwise_matches_dict:
                continue
            match, conf, feature_matches, bp_matches = calc_2frame_matches_using_class(frame0, frame1, use_bipartite_matching)
            pairwise_matches_dict[key] = match
            pairwise_conf_dict[key] = conf
            feature_matches_dict[key] = feature_matches
            if bp_matches is not None:
                bp_matches_dict[key] = list(bp_matches)
    # Use the matches to build a global index
    global2local, local2global = neuron_global_id_from_multiple_matches(
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

    # Build a class to store all the information
    reference_set = RegisteredReferenceFrames(
        global2local,
        local2global,
        ref_frames,
        pairwise_matches,
        pairwise_conf,
        feature_matches,
        bp_matches
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
                                  neuron_feature_radius):
    """
    Multi-frame wrapper around match_to_reference_frames()
    """

    all_matches = []
    all_other_frames = []

    for ind in tqdm(other_ind, total=len(other_ind)):
        dat = get_single_volume(vid_fname, ind, **video_opt)
        metadata['frame_ind'] = ind

        f = build_reference_frame(dat, num_slices, neuron_feature_radius,
                              metadata=metadata)
        matches, _, per_neuron_matches = match_to_reference_frames(f, reference_set)

        all_matches.append(matches)
        f.neuron_ids = per_neuron_matches
        all_other_frames.append(f)

    return all_matches, all_other_frames

##
## Full pipeline function
##

def track_via_reference_frames(vid_fname,
                               start_frame=0,
                               num_frames=10,
                               num_slices=33,
                               alpha=0.15,
                               neuron_feature_radius=5.0,
                               verbose=0,
                               num_reference_frames=5):
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
                 'alpha':alpha,
                 'neuron_feature_radius':neuron_feature_radius,
                 'verbose':verbose-1}
    ref_dat, ref_frames, other_ind = build_all_reference_frames(num_reference_frames, **video_opt)

    if verbose >= 1:
        print("Analyzing reference frames...")
    reference_set = register_all_reference_frames(ref_frames)

    if verbose >= 1:
        print("Matching other frames to reference...")
    video_opt = {'num_slices':num_slices,
                 'alpha':alpha}
    metadata = ref_frames[0].get_metdata()
    all_matches, all_other_frames = match_all_to_reference_frames(
        reference_set,
        vid_fname,
        other_ind,
        video_opt,
        metadata,
        num_slices,
        neuron_feature_radius
    )

    return all_matches, all_other_frames, reference_set


def track_via_sequence_consensus(vid_fname,
                                 start_frame=0,
                                 num_frames=10,
                                 num_slices=33,
                                 alpha=0.15,
                                 neuron_feature_radius=5.0,
                                 verbose=0,
                                 num_consensus_frames=3):
    """
    Tracks neurons by finding consensus between a sliding window of frames

    Note: if num_consensus_frames=2, this is tracking via adjacent frames
    """

    # Initial frame calculations

    # Build a reference set of the first n-1 frames
    video_opt = {'vid_fname':vid_fname,
                 'start_frame':start_frame,
                 'num_frames':num_frames,
                 'num_slices':num_slices,
                 'alpha':alpha,
                 'neuron_feature_radius':neuron_feature_radius,
                 'verbose':verbose-1}
    _, ref_frames, _ = build_all_reference_frames(num_consensus_frames-1, **video_opt)
    reference_set_minus1 = register_all_reference_frames(ref_frames)

    all_frames = ref_frames.copy()
    for i_frame in range():
        # Build the next frame
        frame_video_opt = {'num_slices':num_slices,
                     'alpha':alpha}
        metadata = ref_frames[0].get_metadata()
        next_frame_ind = num_consensus_frames
        dat = get_single_volume(vid_fname, next_frame_ind, **frame_video_opt)
        next_frame = build_reference_frame(dat, num_slices, neuron_feature_radius,
                                           metadata=metadata)
        # Match this frame
        reference_set = register_all_reference_frames(
            [next_frame],
            previous_ref_set=reference_set_minus1
        )

        # Adjust by 1: the new reference set partially overlaps with the previous
        reference_set_minus1 = remove_first_frame(reference_set)

        all_frames.append(next_frame)

    return all_frames
