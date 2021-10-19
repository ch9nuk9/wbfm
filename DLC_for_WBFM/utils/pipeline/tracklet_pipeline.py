import logging
import os
import os.path as osp
import pickle
from pathlib import Path
import pandas as pd

from DLC_for_WBFM.utils.feature_detection.feature_pipeline import track_neurons_full_video
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import build_tracklets_dfs
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig, ConfigFileWithProjectContext, \
    pickle_load_binary
from DLC_for_WBFM.utils.projects.utils_project import get_sequential_filename, safe_cd
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import convert_training_dataframe_to_dlc_format


###
### For use with produces tracklets (step 2 of pipeline)
###
from tqdm.auto import tqdm


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

    raw_fname = training_config.resolve_relative_path(os.path.join('raw', 'clust_df_dat.pickle'), prepend_subfolder=True)
    if os.path.exists(raw_fname):
        raise FileExistsError(f"Found old raw data at {raw_fname}; either rename or skip this step to reuse")

    # Intermediate products: pairwise matches between frames
    video_fname, options = _unpack_config_frame2frame_matches(DEBUG, project_config, training_config)
    all_frame_pairs, all_frame_dict = track_neurons_full_video(video_fname, **options)

    _save_matches_and_frames(all_frame_dict, all_frame_pairs)


def postprocess_and_build_matches_from_config(project_config: ModularProjectConfig,
                                              training_config: ConfigFileWithProjectContext, DEBUG):
    """
    Starting with pairwise matches of neurons between sequential Frame objects, postprocess the matches and generate
    longer tracklets

    Parameters
    ----------
    project_config
    training_config
    DEBUG

    Returns
    -------

    """
    # Load data
    all_frame_dict, all_frame_pairs, z_threshold, min_confidence = \
        _unpack_config_for_tracklets(training_config)

    # Sanity check
    val = len(all_frame_pairs)
    expected = project_config.config['dataset_params']['num_frames'] - 1
    msg = f"Incorrect number of frame pairs ({val} != {expected})"
    assert val == expected, msg

    # Calculate and save in both raw and dataframe format
    df_custom_format = postprocess_and_build_tracklets_from_matches(all_frame_dict, all_frame_pairs, z_threshold, min_confidence)

    # Convert to easier format
    min_length = training_config.config['postprocessing_params']['min_length_to_save']
    df_dlc_format = convert_training_dataframe_to_dlc_format(df_custom_format, min_length=min_length, scorer=None)
    save_all_tracklets(df_custom_format, df_dlc_format, training_config)


def postprocess_and_build_tracklets_from_matches(all_frame_dict, all_frame_pairs, z_threshold, min_confidence, verbose=0):
    # Also updates the matches of the object
    opt = dict(z_threshold=z_threshold, min_confidence=min_confidence)
    logging.info(f"Postprocessing pairwise matches using confidence threshold: {min_confidence}")
    all_matches = {k: pair.calc_final_matches_using_bipartite_matching(**opt)
                   for k, pair in tqdm(all_frame_pairs.items())}
    logging.info("Extracting locations of neurons")
    all_zxy = {k: f.neuron_locs for k, f in tqdm(all_frame_dict.items())}
    logging.info("Building tracklets")
    df = build_tracklets_dfs(all_matches, all_zxy, verbose=verbose)
    return df


def save_all_tracklets(df, df_dlc_format, training_config):
    with safe_cd(training_config.project_dir):
        # Custom format for pairs
        subfolder = osp.join('2-training_data', 'raw')
        fname = osp.join(subfolder, 'clust_df_dat.pickle')
        with open(fname, 'wb') as f:
            pickle.dump(df, f)

        out_fname = training_config.config['df_3d_tracklets']
        df_dlc_format.to_hdf(out_fname, 'df_with_missing')

        out_fname = Path(out_fname).with_suffix(".csv")
        df_dlc_format.to_csv(out_fname)

        # Tracklets are generally too large to save in excel...
        # out_fname = Path(out_fname).with_suffix(".xlxs")
        # training_df.to_excel(out_fname)


def _unpack_config_for_tracklets(training_config):
    params = training_config.config['postprocessing_params']
    z_threshold = params['z_threshold']
    min_confidence = params['min_confidence']

    fname = os.path.join('raw', 'match_dat.pickle')
    fname = training_config.resolve_relative_path(fname, prepend_subfolder=True)
    all_frame_pairs = pickle_load_binary(fname)

    fname = os.path.join('raw', 'frame_dat.pickle')
    fname = training_config.resolve_relative_path(fname, prepend_subfolder=True)
    all_frame_dict = pickle_load_binary(fname)

    return all_frame_dict, all_frame_pairs, z_threshold, min_confidence


def _unpack_config_frame2frame_matches(DEBUG, project_config, training_config):
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

    return video_fname, options


def _save_matches_and_frames(all_frame_dict: dict, all_frame_pairs: dict) -> None:
    subfolder = osp.join('2-training_data', 'raw')
    Path(subfolder).mkdir(exist_ok=True)
    fname = osp.join(subfolder, 'match_dat.pickle')
    [p.prep_for_pickle() for p in all_frame_pairs.values()]
    with open(fname, 'wb') as f:
        pickle.dump(all_frame_pairs, f)
    fname = osp.join(subfolder, 'frame_dat.pickle')
    [frame.prep_for_pickle() for frame in all_frame_dict.values()]
    with open(fname, 'wb') as f:
        pickle.dump(all_frame_dict, f)
