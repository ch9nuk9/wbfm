import logging
import os
import os.path as osp
import pickle
from pathlib import Path
import pandas as pd
from segmentation.util.utils_metadata import DetectedNeurons

from DLC_for_WBFM.utils.feature_detection.class_frame_pair import FramePairOptions
from DLC_for_WBFM.utils.feature_detection.feature_pipeline import track_neurons_full_video, match_all_adjacent_frames
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import build_tracklets_dfs
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from DLC_for_WBFM.utils.projects.utils_filenames import pickle_load_binary
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import convert_training_dataframe_to_scalar_format

###
### For use with produces tracklets (step 2 of pipeline)
###
from tqdm.auto import tqdm


def match_all_adjacent_frames_using_config(project_config: ModularProjectConfig,
                                           training_config: SubfolderConfigFile,
                                           DEBUG: bool = False) -> None:
    """Substep if the frames exist, but the matches are corrupted or need to be redone"""
    logging.info(f"Producing tracklets")

    raw_fname = training_config.resolve_relative_path(os.path.join('raw', 'clust_df_dat.pickle'),
                                                      prepend_subfolder=True)
    if os.path.exists(raw_fname):
        raise FileExistsError(f"Found old raw data at {raw_fname}; either rename or skip this step to reuse")

    frame_fname = training_config.resolve_relative_path(os.path.join('raw', 'frame_dat.pickle'),
                                                        prepend_subfolder=True)
    if not os.path.exists(frame_fname):
        raise FileNotFoundError
    else:
        all_frame_dict = pickle_load_binary(frame_fname)

    # Intermediate products: pairwise matches between frames
    _, tracker_params, pairwise_matches_params = _unpack_config_frame2frame_matches(
        DEBUG, project_config, training_config)
    start_volume = tracker_params['start_volume']
    end_volume = start_volume + tracker_params['num_frames']
    all_frame_pairs = match_all_adjacent_frames(all_frame_dict, end_volume, pairwise_matches_params, start_volume)

    with safe_cd(project_config.project_dir):
        _save_matches_and_frames(all_frame_dict, all_frame_pairs)


def partial_track_video_using_config(project_config: ModularProjectConfig,
                                     training_config: SubfolderConfigFile,
                                     DEBUG: bool = False) -> None:
    """
    Produce training data via partial tracking using 3d feature-based method

    This function is designed to be used with an external .yaml config file

    See new_project_defaults/2-training_data/training_data_config.yaml
    See also track_neurons_full_video()
    """

    logging.info(f"Producing tracklets")

    raw_fname = training_config.resolve_relative_path(os.path.join('raw', 'clust_df_dat.pickle'),
                                                      prepend_subfolder=True)
    if os.path.exists(raw_fname):
        raise FileExistsError(f"Found old raw data at {raw_fname}; either rename or skip this step to reuse")

    # Intermediate products: pairwise matches between frames
    video_fname, tracker_params, pairwise_matches_params = _unpack_config_frame2frame_matches(
        DEBUG, project_config, training_config)
    all_frame_pairs, all_frame_dict = track_neurons_full_video(video_fname, **tracker_params,
                                                               pairwise_matches_params=pairwise_matches_params)
    with safe_cd(project_config.project_dir):
        _save_matches_and_frames(all_frame_dict, all_frame_pairs)


def postprocess_and_build_matches_from_config(project_config: ModularProjectConfig,
                                              segmentation_config: SubfolderConfigFile,
                                              training_config: SubfolderConfigFile, DEBUG):
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
    all_frame_dict, all_frame_pairs, z_threshold, min_confidence, segmentation_metadata = \
        _unpack_config_for_tracklets(training_config, segmentation_config)

    # Sanity check
    val = len(all_frame_pairs)
    expected = project_config.config['dataset_params']['num_frames'] - 1
    msg = f"Incorrect number of frame pairs ({val} != {expected})"
    assert val == expected, msg

    # Calculate and save in both raw and dataframe format
    df_custom_format = postprocess_and_build_tracklets_from_matches(all_frame_dict, all_frame_pairs,
                                                                    z_threshold, min_confidence)
    # Overwrite intermediate products, because the pair objects save the postprocessing options
    with safe_cd(training_config.project_dir):
        _save_matches_and_frames(all_frame_dict, all_frame_pairs)

    # Convert to easier format and save
    min_length = training_config.config['postprocessing_params']['min_length_to_save']
    df_multi_index_format = convert_training_dataframe_to_scalar_format(df_custom_format,
                                                                        min_length=min_length,
                                                                        scorer=None,
                                                                        segmentation_metadata=segmentation_metadata)
    save_all_tracklets(df_custom_format, df_multi_index_format, training_config)


def postprocess_and_build_tracklets_from_matches(all_frame_dict, all_frame_pairs, z_threshold, min_confidence,
                                                 verbose=0):
    # Also updates the matches of the object
    opt = dict(z_threshold=z_threshold, min_confidence=min_confidence)
    logging.info(
        f"Postprocessing pairwise matches using confidence threshold {min_confidence} and z threshold: {z_threshold}")
    all_matches_list = {k: pair.calc_final_matches(**opt)
                        for k, pair in tqdm(all_frame_pairs.items())}
    logging.info("Extracting locations of neurons")
    all_zxy = {k: f.neuron_locs for k, f in all_frame_dict.items()}
    logging.info("Building tracklets")
    return build_tracklets_dfs(all_matches_list, all_zxy, verbose=verbose)


def save_all_tracklets(df, df_multi_index_format, training_config):
    logging.info("Saving dataframes; could take a while")
    with safe_cd(training_config.project_dir):
        # Custom format for pairs
        subfolder = osp.join('2-training_data', 'raw')
        fname = osp.join(subfolder, 'clust_df_dat.pickle')
        with open(fname, 'wb') as f:
            pickle.dump(df, f)

        # General format; ONLY this should be used going forward
        out_fname = training_config.config['df_3d_tracklets']
        df_multi_index_format.to_hdf(out_fname, 'df_with_missing')


def _unpack_config_for_tracklets(training_config, segmentation_config):
    params = training_config.config['pairwise_matching_params']
    z_threshold = params['z_threshold']
    min_confidence = params['min_confidence']
    # matching_method = params['matching_method']

    fname = os.path.join('raw', 'match_dat.pickle')
    fname = training_config.resolve_relative_path(fname, prepend_subfolder=True)
    all_frame_pairs = pickle_load_binary(fname)

    fname = os.path.join('raw', 'frame_dat.pickle')
    fname = training_config.resolve_relative_path(fname, prepend_subfolder=True)
    all_frame_dict = pickle_load_binary(fname)

    seg_metadata_fname = segmentation_config.resolve_relative_path_from_config('output_metadata')
    segmentation_metadata = DetectedNeurons(seg_metadata_fname)

    return all_frame_dict, all_frame_pairs, z_threshold, min_confidence, segmentation_metadata


def _unpack_config_frame2frame_matches(DEBUG, project_config, training_config):
    # Get options
    tracker_params = training_config.config['tracker_params'].copy()
    if 'num_frames' in training_config.config['tracker_params']:
        tracker_params['num_frames'] = training_config.config['tracker_params']['num_frames']
    else:
        tracker_params['num_frames'] = project_config.config['dataset_params']['num_frames']
    if DEBUG:
        tracker_params['num_frames'] = 5
    if 'start_volume' in training_config.config['tracker_params']:
        tracker_params['start_volume'] = training_config.config['tracker_params']['start_volume']
    else:
        tracker_params['start_volume'] = project_config.config['dataset_params']['start_volume']

    pairwise_matches_params = project_config.get_frame_pair_options(training_config)
    tracker_params['preprocessing_settings'] = None

    video_fname = project_config.config['preprocessed_red']

    return video_fname, tracker_params, pairwise_matches_params


def _save_matches_and_frames(all_frame_dict: dict, all_frame_pairs: dict) -> None:
    subfolder = osp.join('2-training_data', 'raw')
    Path(subfolder).mkdir(exist_ok=True)

    fname = osp.join(subfolder, 'frame_dat.pickle')
    [frame.prep_for_pickle() for frame in all_frame_dict.values()]
    with open(fname, 'wb') as f:
        pickle.dump(all_frame_dict, f)

    if all_frame_pairs is not None:
        fname = osp.join(subfolder, 'match_dat.pickle')
        [p.prep_for_pickle() for p in all_frame_pairs.values()]
        with open(fname, 'wb') as f:
            pickle.dump(all_frame_pairs, f)
