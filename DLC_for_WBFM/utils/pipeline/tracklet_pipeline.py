import logging
import os
import os.path as osp
import pickle
from pathlib import Path
import pandas as pd

from DLC_for_WBFM.utils.feature_detection.feature_pipeline import track_neurons_full_video
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import build_tracklets_dfs
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig, ConfigFileWithProjectContext
from DLC_for_WBFM.utils.projects.utils_project import get_sequential_filename, safe_cd
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import convert_training_dataframe_to_dlc_format


###
### For use with produces tracklets (step 2 of pipeline)
###


def partial_track_video_using_config(project_config: ModularProjectConfig,
                                     training_config: ConfigFileWithProjectContext,
                                     DEBUG: bool = False) -> None:
    """
    Produce training data via partial tracking using 3d feature-based method

    This function is designed to be used with an external .yaml config file

    See new_project_defaults/2-training_data/training_data_config.yaml
    See also track_neurons_full_video()
    """
    logging.info(f"Producing tracklets")

    raw_foldername = training_config.resolve_relative_path('raw', prepend_subfolder=True)
    if os.path.exists(raw_foldername):
        raise FileExistsError("Found old raw data folder; either rename or skip this step to reuse")

    video_fname, z_threshold, options = _unpack_config_partial_tracking(DEBUG, project_config, training_config)
    all_frame_pairs, all_frame_dict = track_neurons_full_video(video_fname, **options)

    val = len(all_frame_pairs)
    expected = project_config.config['dataset_params']['num_frames'] - 1
    msg = f"Incorrect number of frame pairs ({val} != {expected})"
    assert val == expected, msg

    df = _postprocess_frame_matches(all_frame_dict, all_frame_pairs, z_threshold)
    save_all_tracklets(all_frame_dict, all_frame_pairs, df, project_config, training_config)


def _postprocess_frame_matches(all_frame_dict, all_frame_pairs, z_threshold=None, verbose=0):
    # Also updates the matches of the object
    all_matches = {k: pair.calc_final_matches_using_bipartite_matching(z_threshold=z_threshold)
                   for k, pair in all_frame_pairs.items()}
    all_zxy = {k: f.neuron_locs for k, f in all_frame_dict.items()}
    df = build_tracklets_dfs(all_matches, all_zxy, verbose=verbose)
    return df


def save_all_tracklets(all_frame_dict, all_frame_pairs, df, project_config, training_config):
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

        # Tracklets are generally too large to save in excel...
        # out_fname = Path(out_fname).with_suffix(".xlxs")
        # training_df.to_excel(out_fname)


def _unpack_config_partial_tracking(DEBUG, project_config, training_config):
    # Make tracklets
    # Get options
    options = training_config.config['tracker_params'].copy()
    if 'num_frames' in training_config.config['tracker_params']:
        options['num_frames'] = training_config.config['tracker_params']['num_frames']
    else:
        options['num_frames'] = project_config.config['dataset_params']['num_frames']
    if DEBUG:
        options['num_frames'] = 5
    if 'start_volume' in training_config.config['tracker_params']:
        options['start_volume'] = training_config.config['tracker_params']['start_volume']
    else:
        options['start_volume'] = project_config.config['dataset_params']['start_volume']
    options['num_slices'] = project_config.config['dataset_params']['num_slices']

    options['preprocessing_settings'] = None

    video_fname = project_config.config['preprocessed_red']
    z_threshold = training_config.config['postprocessing_params']['z_threshold']

    return video_fname, z_threshold, options


def _save_matches_and_frames(all_frame_dict: dict, all_frame_pairs: dict, df: pd.DataFrame) -> None:
    subfolder = osp.join('2-training_data', 'raw')
    subfolder = get_sequential_filename(subfolder)
    os.mkdir(subfolder)
    fname = osp.join(subfolder, 'clust_df_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(df, f)
    fname = osp.join(subfolder, 'match_dat.pickle')
    [p.prep_for_pickle() for p in all_frame_pairs.values()]
    with open(fname, 'wb') as f:
        pickle.dump(all_frame_pairs, f)
    fname = osp.join(subfolder, 'frame_dat.pickle')
    [frame.prep_for_pickle() for frame in all_frame_dict.values()]
    with open(fname, 'wb') as f:
        pickle.dump(all_frame_dict, f)
