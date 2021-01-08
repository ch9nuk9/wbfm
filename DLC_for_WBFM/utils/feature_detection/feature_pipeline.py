from DLC_for_WBFM.utils.feature_detection.utils_features import *
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import *
from DLC_for_WBFM.utils.feature_detection.utils_detection import *
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
import copy
import numpy as np
import time
from tqdm import tqdm
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
    keypoints: list
    keypoint_locs: list # Just the z coordinate
    all_features: np.array
    features_to_neurons: list
    neuron_ids: list = None # global neuron index

    # Metadata
    frame_ind: int = None

    def iter_neurons(self):
        # Practice with yield
        for neuron in self.neuron_locs:
            yield neuron

    def get_features_of_neuron(self, which_neuron):
        return np.where(self.features_to_neurons == which_neuron)

    def num_neurons(self):
        return self.neuron_locs.shape[0]


def build_reference_frames(num_reference_frames,
                         vid_fname,
                         start_frame,
                         num_frames,
                         num_slices,
                         neuron_feature_radius,
                         alpha,
                         verbose=1):
    """
    Selects a sample of reference frames, then builds features for them
    """

    other_ind = list(range(start_frame, start_frame+num_frames))
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
        kps, kp_3d_locs, features = build_features_1volume(dat, num_features_per_plane=1000)

        # The map requires some open3d subfunctions
        num_f, pc_f, _ = build_feature_tree(kp_locs, which_slice=None)
        _, _, tree_neurons = build_neuron_tree(neuron_locs, to_mirror=False)
        f2n_map = build_f2n_map(kp_3d_locs,
                               num_f,
                               pc_f,
                               neuron_feature_radius,
                               tree_neurons,
                               verbose=0)

        # Finally, my summary class
        f = ReferenceFrame(neuron_locs, kps, kp_3d_locs, features, f2n_map, None, ind)
        ref_frames.append(f)

    return ref_dat, ref_frames, other_ind


def calc_2frame_matches_using_class(frame0,
                                    frame1,
                                    verbose=1):
    """
    Similar to older function, but this doesn't assume the features are
    already matched

    See also: calc_2frame_matches
    """

    # First, get feature matches
    matches = match_known_features(frame0.all_features,
                                   frame1.all_features,
                                   frame0.keypoints,
                                   frame1.keypoints)
    # TODO: is this a single list?

    # Second, get neuron matches
    all_neuron_matches = []
    all_confidences = []
    for i, neuron in frame0.iter_neurons():
        # Get features of this neuron
        this_f0 = frame0.get_features_of_neuron(i)
        # Use matches to translate to the indices of frame1
        this_f1 = feature_matches[this_f0]
        # Get the corresponding neurons in vol1, and vote
        this_n1 = features_to_neurons1[this_f1]

        all_neuron_matches, all_confidences = add_neuron_match(
            all_neuron_matches,
            all_confidences,
            i,
            this_n1,
            verbose
        )

    return all_neuron_matches, all_confidences


def register_all_reference_frames(ref_frames, verbose=1):
    """
    Registers a set of reference frames, aligning their neuron indices

    Builds all
    """

    ref_neuron_ind = []
    if verbose >= 1:
        print("Pairwise matching all reference frames...")
    for i0, frame0 in tqdm(enumerate(ref_frames), total=len(ref_frames)):
        matches_this_frame = []
        for i1, frame1 in enumerate(ref_frames):
            if i1==i0:
                continue
            matches_this_frame.append(calc_2frame_matches_using_class(frame0, frame1))
        # TODO: actually use the matches

    return ref_neuron_ind


def match_to_reference_frames(this_frame, ref_frames):
    """
    Registers a single frame to a set of references
    """

    matches = []

    return matches


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
    ref_dat, ref_frames, other_ind = build_reference_frames(num_reference_frames, **video_opt)

    # dataframe with features and feature-ind dict (separated by ref frame)
    if verbose >= 1:
        print("Analyzing reference frames...")
    ref_frames = register_all_reference_frames(ref_frames)

    if verbose >= 1:
        print("Matching other frames to reference...")
    video_opt = {'num_slices':num_slices,
                 'alpha':alpha}
    all_matches = []
    for ind in other_ind:
        break
        this_frame = get_single_volume(vid_fname, ind, **video_opt)
        matches = match_to_reference_frames(this_frame, ref_frames)
        all_matches.append(matches)

    return ref_frames
