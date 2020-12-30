from DLC_for_WBFM.utils.feature_detection.utils_features import *
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import *
from DLC_for_WBFM.utils.feature_detection.utils_detection import *
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
import copy
import numpy as np


##
## Full pipeline
##

def track_neurons_two_volumes(dat0,
                              dat1,
                              num_slices=33,
                              verbose=1):
    """
    Matches neurons between two volumes
    """
    # Detect neurons, then features for each volume
    opt = {'num_slices':num_slices,
           'alpha':1.0, # Already multiplied when imported
           'verbose':verbose-1}
    neurons0, _, _, _ = detect_neurons_using_ICP(dat0, **opt)
    neurons1, _, _, _ = detect_neurons_using_ICP(dat1, **opt)

    opt = {'verbose':verbose-1,
           'matches_to_keep':0.8,
           'num_features_per_plane':10000,
           'detect_keypoints':True,
           'kp0':neurons0,
           'kp1':neurons1}
    all_f0, all_f1, _, _ = build_features_on_all_planes(dat0,dat1,**opt)

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
    return all_matches, all_conf


def track_neurons_full_video(vid_fname,
                             start_frame=0,
                             num_frames=10,
                             num_slices=33,
                             alpha=0.15,
                             verbose=0):
    """
    Detects and tracks neurons using opencv-based feature matching
    """

    # Get initial volume; settings are same for all
    import_opt = {'num_slices':num_slices, 'alpha':alpha}
    dat0 = get_single_volume(vid_fname, start_frame, **import_opt)

    # Loop through all pairs
    all_matches = []
    all_conf = []
    end_frame = start_frame+num_frames
    frame_range = range(start_frame+1, end_frame)
    for i_frame in frame_range:
        if verbose >= 1:
            print("===========================================================")
            print(f"Matching frames {i_frame-1} and {i_frame} (end at {end_frame})")
        dat1 = get_single_volume(vid_fname, i_frame, **import_opt)

        matches, conf = track_neurons_two_volumes(dat0,
                                                  dat1,
                                                  num_slices=num_slices,
                                                  verbose=verbose-1)
        all_matches.append(matches)
        all_conf.append(conf)

        dat0 = copy.copy(dat1)

    return all_matches, all_conf
