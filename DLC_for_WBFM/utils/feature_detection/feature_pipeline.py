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
                             verbose=0):
    """
    Detects and tracks neurons using opencv-based feature matching
    """
    start_time = time.time()

    # Get initial volume; settings are same for all
    import_opt = {'num_slices':num_slices, 'alpha':alpha}
    dat0 = get_single_volume(vid_fname, start_frame, **import_opt)

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

def build_reference_frames(num_reference_frames,
                         vid_fname,
                         start_frame,
                         num_frames,
                         num_slices,
                         neuron_feature_radius,
                         alpha,
                         is_sequential=True,
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

        # Get neurons and features, and a map between them
        neuron_locs, _, _, icp_kps = detect_neurons_using_ICP(dat,
                                                             num_slices=num_slices,
                                                             alpha=1.0,
                                                             min_detections=3,
                                                             verbose=0)
        neuron_locs = np.array([n for n in neuron_locs])
        feature_opt = {'num_features_per_plane':1000, 'start_plane':5}
        kps, kp_3d_locs, features = build_features_1volume(dat, **feature_opt)

        # The map requires some open3d subfunctions
        num_f, pc_f, _ = build_feature_tree(kp_3d_locs, which_slice=None)
        _, _, tree_neurons = build_neuron_tree(neuron_locs, to_mirror=False)
        #zzz
        f2n_map = build_f2n_map(kp_3d_locs,
                               num_f,
                               pc_f,
                               neuron_feature_radius,
                               tree_neurons,
                               verbose=verbose-1)

        # Finally, my summary class
        metadata = {'frame_ind':ind,
                    'vol_shape':dat.shape,
                    'video_fname':vid_fname,
                    'alpha':alpha}
        f = ReferenceFrame(neuron_locs, kps, kp_3d_locs, features, f2n_map, **metadata)
        ref_frames.append(f)

    return ref_dat, ref_frames, other_ind


def calc_2frame_matches_using_class(frame0,
                                    frame1,
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

        min_features_needed = 5 # TODO
        all_neuron_matches, all_confidences = add_neuron_match(
            all_neuron_matches,
            all_confidences,
            neuron0_ind,
            5,
            this_n1,
            this_f1,
            verbose-1
        )
        if DEBUG:
            break

    return all_neuron_matches, all_confidences, feature_matches


##
## Networkx-based construction of reference indices
##

def neuron_global_id_from_multiple_matches(pairwise_matches_dict):
    """
    Builds a vector of neuron matches from pairwise matchings to multiple frames

    Input format:
        pairwise_matches_dict[T] -> match_array
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
    return

##
## Matching the features of the frames
##


def register_all_reference_frames(ref_frames, verbose=1):
    """
    Registers a set of reference frames, aligning their neuron indices

    Builds all
    """

    ref_neuron_ind = []
    pairwise_matches_dict = {}
    feature_matches_dict = {}
    pairwise_conf_dict = {}
    if verbose >= 1:
        print("Pairwise matching all reference frames...")
    for i0, frame0 in tqdm(enumerate(ref_frames), total=len(ref_frames)):
        for i1, frame1 in enumerate(ref_frames):
            if i1==i0:
                continue
            match, conf, feature_matches = calc_2frame_matches_using_class(frame0, frame1)
            key = (i0, i1)
            pairwise_matches_dict[key] = match
            pairwise_conf_dict[key] = conf
            feature_matches_dict[key] = feature_matches
    # TODO: Use the matches to be build a global index
    global_neuron_ind = neuron_global_id_from_multiple_matches(pairwise_matches_dict)

    return global_neuron_ind, pairwise_matches_dict, pairwise_conf_dict, feature_matches_dict


def match_to_reference_frames(this_frame, ref_frames):
    """
    Registers a single frame to a set of references
    """

    matches = []

    return matches


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
    ref_dat, ref_frames, other_ind = build_reference_frames(num_reference_frames, **video_opt)

    # dataframe with features and feature-ind dict (separated by ref frame)
    if verbose >= 1:
        print("Analyzing reference frames...")
    ref_neuron_ind, pairwise_matches, pairwise_conf, feature_matches = register_all_reference_frames(ref_frames)

    if verbose >= 1:
        print("Matching other frames to reference...")
    video_opt = {'num_slices':num_slices,
                 'alpha':alpha}
    all_matches = []
    for ind in other_ind:
        break # WIP
        this_frame = get_single_volume(vid_fname, ind, **video_opt)
        matches = match_to_reference_frames(this_frame, ref_frames)
        all_matches.append(matches)

    return ref_frames, all_matches, pairwise_matches, pairwise_conf, feature_matches
