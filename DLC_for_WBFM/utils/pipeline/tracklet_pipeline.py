from DLC_for_WBFM.utils.feature_detection.feature_pipeline import track_neurons_full_video
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
from DLC_for_WBFM.utils.feature_detection.utils_candidate_matches import fix_candidates_without_confidences, calc_all_bipartite_matches
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import build_tracklets_from_classes
from DLC_for_WBFM.utils.projects.utils_project import get_sequential_filename

import os
import os.path as osp
import numpy as np
import pandas as pd
import pickle

###
### For use with produces tracklets (step 2 of pipeline)
###

def partial_track_video_using_config(vid_fname, config, DEBUG=False):
    """
    Produce training data via partial tracking using 3d feature-based method

    This function is designed to be used with an external .yaml config file

    See new_project_defaults/2-training_data/training_data_config.yaml
    See also track_neurons_full_video()
    """

    # Load preprocessing settings
    p_fname = config['preprocessing_config']
    p = PreprocessingSettings.load_from_yaml(p_fname)

    ########################
    # Make tracklets
    ########################
    # Get options
    opt = config['tracker_params'].copy()
    opt['num_frames'] = config['dataset_params']['num_frames']
    if DEBUG:
        opt['num_frames'] = 5
    opt['start_frame'] = config['dataset_params']['start_volume']
    opt['num_slices'] = config['dataset_params']['num_slices']

    out = track_neurons_full_video(vid_fname,
                                   preprocessing_settings=p,
                                   **opt)
    ########################
    # Postprocess matches
    ########################
    b_matches, b_conf, b_frames, b_candidates = out
    new_candidates = fix_candidates_without_confidences(b_candidates)
    bp_matches = calc_all_bipartite_matches(new_candidates)
    df = build_tracklets_from_classes(b_frames, bp_matches)

    ########################
    # Save matches to disk
    ########################
    subfolder = osp.join('2-training_data', 'raw')
    subfolder = get_sequential_filename(subfolder)
    os.mkdir(subfolder)

    fname = osp.join(subfolder, 'clust_df_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(df,f)
    fname = osp.join(subfolder, 'match_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(b_matches, f)
    fname = osp.join(subfolder, 'candidate_matches_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(new_candidates, f)
    fname = osp.join(subfolder, 'frame_dat.pickle')
    [frame.prep_for_pickle() for frame in b_frames.values()]
    with open(fname, 'wb') as f:
        pickle.dump(b_frames, f)
