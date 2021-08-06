from pathlib import Path

from DLC_for_WBFM.utils.feature_detection.feature_pipeline import track_neurons_full_video
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
from DLC_for_WBFM.utils.feature_detection.utils_candidate_matches import fix_candidates_without_confidences, calc_all_bipartite_matches
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import build_tracklets_from_classes, build_tracklets_dfs
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config, config_file_with_project_context
from DLC_for_WBFM.utils.projects.utils_project import get_sequential_filename, safe_cd

import os
import os.path as osp
import numpy as np
import pandas as pd
import pickle

###
### For use with produces tracklets (step 2 of pipeline)
###


# def partial_track_video_using_config(vid_fname: str, config: dict, DEBUG: bool = False) -> None:
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import save_all_tracklets_as_dlc_format, \
    convert_training_dataframe_to_dlc_format


def partial_track_video_using_config(project_config: modular_project_config,
                                     training_config: config_file_with_project_context, DEBUG: bool = False) -> None:
    """
    Produce training data via partial tracking using 3d feature-based method

    This function is designed to be used with an external .yaml config file

    See new_project_defaults/2-training_data/training_data_config.yaml
    See also track_neurons_full_video()
    """

    vid_fname, opt = _unpack_config_partial_tracking(DEBUG, project_config, training_config)
    all_frame_pairs, all_frame_dict = track_neurons_full_video(vid_fname, **opt)

    # Also updates the matches of the object
    all_matches = {k: pair.calc_final_matches_using_bipartite_matching() for k, pair in all_frame_pairs.items()}

    all_xyz = {k: f.neuron_locs for k, f in all_frame_dict.items()}
    df = build_tracklets_dfs(all_matches, all_xyz)

    with safe_cd(project_config.project_dir):
        # Intermediate products
        _save_matches_and_frames(all_frame_dict, all_frame_pairs, df)

        # Postprocess and save final output
        min_length = training_config.config['postprocessing_params']['min_length_to_save']
        training_df = convert_training_dataframe_to_dlc_format(df, min_length=min_length, scorer=None)

        out_fname = training_config.config['df_raw_3d_tracks']
        training_df.to_hdf(out_fname, 'df_with_missing')

        out_fname = Path(out_fname).with_suffix(".csv")
        training_df.to_csv(out_fname)


def _unpack_config_partial_tracking(DEBUG, project_config, training_config):
    # Make tracklets
    # Get options
    opt = training_config.config['tracker_params'].copy()
    if 'num_frames' in training_config.config['tracker_params']:
        opt['num_frames'] = training_config.config['tracker_params']['num_frames']
    else:
        opt['num_frames'] = project_config.config['dataset_params']['num_frames']
    if DEBUG:
        opt['num_frames'] = 5
    if 'start_volume' in training_config.config['tracker_params']:
        opt['start_volume'] = training_config.config['tracker_params']['start_volume']
    else:
        opt['start_volume'] = project_config.config['dataset_params']['start_volume']
    opt['num_slices'] = project_config.config['dataset_params']['num_slices']

    opt['preprocessing_settings'] = None

    vid_fname = project_config.config['preprocessed_red']

    return vid_fname, opt


def _save_matches_and_frames(all_frame_dict: dict, all_frame_pairs: dict, df: pd.DataFrame) -> None:
    subfolder = osp.join('2-training_data', 'raw')
    subfolder = get_sequential_filename(subfolder)
    os.mkdir(subfolder)
    fname = osp.join(subfolder, 'clust_df_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(df, f)
    fname = osp.join(subfolder, 'match_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(all_frame_pairs, f)
    fname = osp.join(subfolder, 'frame_dat.pickle')
    [frame.prep_for_pickle() for frame in all_frame_dict.values()]
    with open(fname, 'wb') as f:
        pickle.dump(all_frame_dict, f)
