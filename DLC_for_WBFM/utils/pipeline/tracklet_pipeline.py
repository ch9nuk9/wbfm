from DLC_for_WBFM.utils.feature_detection.visualize_using_dlc import training_data_from_annotations
from DLC_for_WBFM.utils.feature_detection.utils_pipeline import track_neurons_full_video
# from DLC_for_WBFM.utils.projects.utils_project import load_config
from DLC_for_WBFM.utils.feature_detection.utils_tif import PreprocessingSettings

import os
import os.path as osp


def partial_track_video_using_config(vid_fname, _config, scorer=None):
    """
    Produce training data via partial tracking using 3d feature-based method

    This function is designed to be used with an external .yaml config file

    See new_project_defaults/2-training_data/training_data_config.yaml
    See also track_neurons_full_video()
    """

    # Load preprocessing settings
    p_fname = _config['preprocessing_config']
    p = PreprocessingSettings.load_from_yaml(p_fname)

    ########################
    # Make tracklets
    ########################
    # Get options
    opt = _config['tracker_params']
    opt['num_frames'] = _config['dataset_params']['num_frames']
    opt['start_frame'] = _config['dataset_params']['start_volume']
    opt['num_slices'] = _config['dataset_params']['num_slices']

    out = track_neurons_full_video(vid_fname,
                                   preprocessing_settings=p,
                                   **opt)
    ########################
    # Postprocess matches
    ########################
    b_matches, b_conf, b_frames, b_candidates = out
    # align gaussian process formatting
    new_candidates = {}
    for k, these_matches in b_candidates.items():
        new_matches = []
        for m in these_matches:
            if len(m)==3:
                m = (int(m[0]), int(m[1]), m[2])
            else:
                m = (int(m[0]), int(m[1]), 1.0)
            new_matches.append(m)
        new_candidates[k] = new_matches

    bp_matches = calc_all_bipartite_matches(new_candidates)
    df_b = build_tracklets_from_classes(b_frames, bp_matches)

    ########################
    # Save matches to disk
    ########################
    subfolder = osp.join('2-training_data', 'raw')
    os.mkdir(subfolder)

    fname = osp.join(subfolder, 'clust_df_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(df_b,f)
    fname = osp.join(subfolder, 'match_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(b_matches, f)
    fname = osp.join(subfolder, 'frame_dat.pickle')
    [frame.prep_for_pickle() for frame in b_frames.values()]
    with open(fname, 'wb') as f:
        pickle.dump(b_frames, f)

    ########################
    # Make dlc-style training data
    ########################

    opt = {}
    opt['df_fname'] = osp.join(subfolder, 'clust_df_dat.pickle')
    opt['scorer'] = scorer
    opt['total_num_frames'] = _config['dataset_params']['num_frames']
    opt['coord_names'] = ['x','y','likelihood']
    # TODO
    opt['which_frames'] = get_good_frames_for_tracklets()
    # TODO: refactor away from old-style config
    new_dlc_df = training_data_from_annotations(config, **opt)
