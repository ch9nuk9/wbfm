import logging
import os
import os.path as osp
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from segmentation.util.utils_metadata import DetectedNeurons

from DLC_for_WBFM.utils.external.utils_pandas import get_names_from_df
from DLC_for_WBFM.utils.neuron_matching.feature_pipeline import track_neurons_full_video, match_all_adjacent_frames
from DLC_for_WBFM.utils.projects.utils_neuron_names import name2int_neuron_and_tracklet, int2name_tracklet
from DLC_for_WBFM.utils.tracklets.utils_tracklets import build_tracklets_dfs, split_multiple_tracklets
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from DLC_for_WBFM.utils.projects.utils_filenames import pickle_load_binary
from DLC_for_WBFM.utils.projects.utils_project import safe_cd

###
### For use with produces tracklets (step 2 of traces)
###
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.tracklets.tracklet_to_DLC import convert_training_dataframe_to_scalar_format


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
    segmentation_config
    project_config
    training_config
    DEBUG

    Returns
    -------

    """
    # Load data
    all_frame_dict, all_frame_pairs, z_threshold, min_confidence, segmentation_metadata, postprocessing_params = \
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
    min_length = postprocessing_params['min_length_to_save']
    df_multi_index_format = convert_training_dataframe_to_scalar_format(df_custom_format,
                                                                        min_length=min_length,
                                                                        scorer=None,
                                                                        segmentation_metadata=segmentation_metadata)
    if postprocessing_params.get('volume_percent_threshold', 0) > 0:
        df_multi_index_format, split_times = filter_tracklets_using_volume(df_multi_index_format,
                                                                           **postprocessing_params)
        out_fname = '2-training_data/raw/volume_tracklet_split_points.pickle'
        training_config.pickle_in_local_project(split_times, out_fname)

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

    postprocessing_params = training_config.config['postprocessing_params']

    return all_frame_dict, all_frame_pairs, z_threshold, min_confidence, segmentation_metadata, postprocessing_params


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

    metadata_fname = tracker_params['external_detections']
    tracker_params['external_detections'] = training_config.resolve_relative_path(metadata_fname)

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
    else:
        logging.warning(f"all_frame_pairs is None; this step will need to be rerun")


def filter_tracklets_using_volume(df_all_tracklets, volume_percent_threshold, min_length_to_save, verbose=0,
                                  DEBUG=False):
    """
    Split the tracklets based on a threshold on the percentage change in volume

    Usually, if the volume changes by a lot, it is because there is a segmentation error
    """
    # Get the split points
    df_only_volume = df_all_tracklets.xs('volume', level=1, axis=1)
    df_percent_changes = df_only_volume.diff() / df_only_volume

    df_split_points = df_percent_changes.abs() > volume_percent_threshold
    t_split_points, i_tracklet_split_points = df_split_points.to_numpy().nonzero()

    # Reformat the split points to be a dict per-tracklet
    all_names = get_names_from_df(df_only_volume)
    tracklet2split = defaultdict(list)
    for t, i_tracklet in zip(t_split_points, i_tracklet_split_points):
        tracklet_name = all_names[i_tracklet]
        tracklet2split[tracklet_name].append(t)

    # Get all the candidate tracklets, including the raw ones if no split detected
    all_new_tracklets = []
    all_names.sort()
    i_next_name = name2int_neuron_and_tracklet(all_names[-1])
    if verbose >= 1:
        print(f"New tracklets starting at index: {i_next_name + 1}")
    # convert_to_sparse = lambda x: pd.arrays.SparseArray(np.squeeze(x.values))
    for name in tqdm(all_names, leave=False):
        this_tracklet = df_all_tracklets[[name]]
        # this_tracklet.loc[name] = this_tracklet.groupby(level=1, axis=1).apply(convert_to_sparse)
        if name in tracklet2split:
            split_points = tracklet2split[name]
            these_candidates = split_multiple_tracklets(this_tracklet, split_points)
            # Remove short ones, and rename
            these_candidates = [c for c in these_candidates if c[name]['z'].count() >= min_length_to_save]
            for i, c in enumerate(these_candidates):
                if i == 0:
                    # The first tracklet keeps the original name
                    all_new_tracklets.append(c)
                    continue
                i_next_name += 1
                all_new_tracklets.append(c.rename(mapper={name: int2name_tracklet(i_next_name)}, axis=1))

        else:
            all_new_tracklets.append(this_tracklet)
        if DEBUG:
            print(tracklet2split[name])
            print(all_new_tracklets)
            break

    if verbose >= 1:
        print(f"Split {len(all_names)} raw tracklets into {len(all_new_tracklets)} new tracklets")
        print("Now concatenating...")

    # Convert to sparse datatype
    # all_converted_tracklets = [t.groupby(level=1, axis=1).apply(convert_to_sparse) for t in tqdm(all_new_tracklets, leave=False)]

    # Remake original all-tracklet dataframe
    # df = pd.concat(all_converted_tracklets)
    df = pd.concat(all_new_tracklets, axis=1)
    if verbose >= 1:
        print("Finished")
    return df, tracklet2split
