from DLC_for_WBFM.utils.feature_detection.utils_features import *
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import *
from DLC_for_WBFM.utils.feature_detection.utils_detection import *
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
import copy
import numpy as np
import time
import tqdm
import random
from dataclasses import dataclass

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

@dataclass
class ReferenceFrame():
    """ Information for registered reference frames"""

    # Data for registration
    neuron_locs: list
    all_features: list
    features_to_neurons: list
    neuron_ids: list = None # global neuron index

    # Metadata
    frame_ind: int = None


def get_reference_frames(num_reference_frames,
                         vid_fname,
                         start_frame,
                         num_frames,
                         num_slices,
                         neuron_feature_radius,
                         alpha):
    """
    Selects a sample of reference frames, then builds features for them
    """

    frame_range = range(start_frame, start_frame+num_frames)
    ref_ind = random.sample(frame_range, num_reference_frames)

    ref_dat = []
    ref_frames = []
    video_opt = {'num_slices':num_slices,
                 'alpha':alpha}
    for ind in ref_ind:
        dat = get_single_volume(vid_fname, ind, **video_opt)
        ref_dat.append(dat)

        # Get neurons and features, and a map between them
        neuron_locs, _, _, icp_kps = detect_neurons_using_ICP(dat,
                                                             num_slices=num_slices,
                                                             alpha=1.0,
                                                             min_detections=3,
                                                             verbose=0)
        kp_locs, features = build_features_1volume(dat, num_features_per_plane=1000)

        # The map requires some open3d subfunctions
        num_f, pc_f, _ = build_feature_tree(kp_locs, which_slice=None)
        _, _, tree_neurons = build_neuron_tree(neuron_locs, to_mirror=False)
        f2n_map = build_f2n_map(kp_locs,
                               num_f,
                               pc_f,
                               neuron_feature_radius,
                               tree_neurons,
                               verbose=0)

        # Finally, my summary class
        ref_frames.append(ReferenceFrame(neurons, features, f2n_map, None, ind))

    return ref_dat, ref_frames


def register_reference_frames(ref_frames, ref_dat):
    """
    Registers a set of reference frames, aligning their neuron indices
    """
    print("WIP")

    #for ind, dat in zip(ref_dat, ref_ind):



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
                 'neuron_feature_radius':neuron_feature_radius}
    ref_dat, ref_frames = get_reference_frames(num_reference_frames, **video_opt)

    # dataframe with features and feature-ind dict (separated by ref frame)
    if verbose >= 1:
        print("Analyzing reference frames...")
    #ref_results = register_reference_frames(ref_dat, ref_ind, num_slices)

    if verbose >= 1:
        print("Matching other frames to reference...")
