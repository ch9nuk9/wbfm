import logging
import os
import os.path as osp
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from segmentation.util.utils_metadata import DetectedNeurons

from DLC_for_WBFM.utils.external.utils_pandas import get_names_from_df, check_if_heterogenous_columns
from DLC_for_WBFM.utils.neuron_matching.feature_pipeline import track_neurons_full_video, match_all_adjacent_frames, \
    calculate_frame_objects_full_video
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.projects.utils_neuron_names import name2int_neuron_and_tracklet, int2name_tracklet
from DLC_for_WBFM.utils.tracklets.high_performance_pandas import delete_tracklets_using_ground_truth, PaddedDataFrame
from DLC_for_WBFM.utils.tracklets.utils_tracklets import build_tracklets_dfs, split_multiple_tracklets, \
    get_next_name_generator, remove_tracklets_from_dictionary_without_database_match
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


def build_frame_objects_using_config(project_config: ModularProjectConfig,
                                     training_config: SubfolderConfigFile,
                                     DEBUG: bool = False) -> None:
    """
    Produce (or rebuild) the Frame objects associated with a project, without matching them or otherwise producing
    tracklets

    This function is designed to be used with an external .yaml config file

    See new_project_defaults/2-training_data/training_data_config.yaml
    See also partial_track_video_using_config()
    """
    logging.info(f"Producing per-volume ReferenceFrame objects")
    video_fname, tracker_params, _ = _unpack_config_frame2frame_matches(DEBUG, project_config, training_config)

    dtype = 'uint8'

    # Build frames, then match them
    tracker_params['end_volume'] = tracker_params['start_volume'] + tracker_params['num_frames']
    all_frame_dict = calculate_frame_objects_full_video(video_fname=video_fname, **tracker_params)
    with safe_cd(project_config.project_dir):
        _save_matches_and_frames(all_frame_dict, None)


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
    volume_threshold = postprocessing_params.get('volume_percent_threshold', 0)
    if volume_threshold > 0:
        logging.info(f"Postprocessing using volume threshold: {volume_threshold}")
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
        # Update to save as sparse from the beginning
        out_fname = training_config.config['df_3d_tracklets']
        # df_multi_index_format.to_hdf(out_fname, 'df_with_missing')
        logging.info("Converting dataframe to sparse format")
        df_multi_index_format = df_multi_index_format.astype(pd.SparseDtype("float", np.nan))
        training_config.pickle_in_local_project(df_multi_index_format, out_fname, custom_writer=pd.to_pickle)


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


def split_tracklets_using_change_detection(project_cfg: ModularProjectConfig, DEBUG=False):
    # For now, do not update any global2tracklet matches; assume this is done before step 3b
    project_data = ProjectData.load_final_project_data_from_config(project_cfg, to_load_tracklets=True)
    initial_empty_cols = 500000
    if DEBUG:
        initial_empty_cols = 10

    # Unpack
    # training_cfg = project_cfg.get_training_config()
    df_tracklets = project_data.df_all_tracklets
    # df_gt = project_data.final_tracks
    # sanity_checks_on_dataframes(df_gt, df_tracklets)
    # g2t = project_data.global2tracklet

    logging.info("Splitting jumping tracklets using custom dataframe class")
    df_padded = PaddedDataFrame.construct_from_basic_dataframe(df_tracklets, name_mode='tracklet',
                                                               initial_empty_cols=initial_empty_cols)
    df_split, name_mapping = df_padded.split_all_tracklets_using_mode(split_mode='jump', verbose=0)
    # global2tracklet_new = update_global2tracklet_dictionary(df_split, g2t, name_mapping)

    df_final = df_split.return_sparse_dataframe()

    # Save and update configs
    training_cfg = project_cfg.get_training_config()
    out_fname = os.path.join('2-training_data', 'all_tracklets_after_splitting.pickle')
    training_cfg.pickle_in_local_project(df_final, relative_path=out_fname, custom_writer=pd.to_pickle)

    # global2tracklet_matches_fname = os.path.join('3-tracking', 'global2tracklet_after_splitting.pickle')
    # tracking_cfg.pickle_in_local_project(global2tracklet_new, global2tracklet_matches_fname)
    # tracking_cfg.config['global2tracklet_matches_fname'] = global2tracklet_matches_fname
    training_cfg.config['df_3d_tracklets'] = out_fname
    training_cfg.update_on_disk()
    # tracking_cfg.update_on_disk()


def overwrite_tracklets_using_ground_truth(project_cfg: ModularProjectConfig,
                                           keep_new_tracklet_matches=False,
                                           update_only_finished_neurons=False, DEBUG=False):
    project_data = ProjectData.load_final_project_data_from_config(project_cfg, to_load_tracklets=True)

    # Unpack
    # df_tracklets = project_data.df_all_tracklets
    # Get the tracklets directly from step 2
    training_cfg = project_cfg.get_training_config()
    tracking_cfg = project_cfg.get_tracking_config()
    fname = training_cfg.resolve_relative_path_from_config('df_3d_tracklets')
    df_tracklets = pd.read_pickle(fname)
    df_gt = project_data.final_tracks
    sanity_checks_on_dataframes(df_gt, df_tracklets)

    if update_only_finished_neurons:
        neurons_that_are_finished, _ = project_data.get_ground_truth_annotations()
    else:
        logging.info("Assuming partially tracked neurons are correct")
        neurons_that_are_finished = None

    # Delete conflicting tracklets, then concat
    df_tracklets_no_conflict, _, _ = delete_tracklets_using_ground_truth(df_gt, df_tracklets,
                                                                   gt_names=neurons_that_are_finished,
                                                                   DEBUG=DEBUG)

    if neurons_that_are_finished is not None:
        df_to_concat = df_gt.loc[:, neurons_that_are_finished]
    else:
        df_to_concat = df_gt
    neuron_names = get_names_from_df(df_to_concat)
    name_gen = get_next_name_generator(df_tracklets_no_conflict)
    gtneuron2tracklets = {name: new_name for name, new_name in zip(neuron_names, name_gen)}
    df_to_concat = df_to_concat.rename(mapper=gtneuron2tracklets, axis=1)

    logging.info("Large pandas concat, may take a while...")
    df_including_tracks = pd.concat([df_tracklets_no_conflict, df_to_concat], axis=1)

    logging.info("Splitting non-contiguous tracklets using custom dataframe class")
    df_padded = PaddedDataFrame.construct_from_basic_dataframe(df_including_tracks, name_mode='tracklet',
                                                               initial_empty_cols=10000)
    df_split, name_mapping = df_padded.split_all_tracklets_using_mode(split_mode='gap', verbose=0)

    # Keep the names as they are in the ground truth track
    global2tracklet_new = update_global2tracklet_dictionary(df_split, gtneuron2tracklets, name_mapping)

    df_final = df_split.return_sparse_dataframe()

    if keep_new_tracklet_matches:
        raise NotImplementedError
        # TODO: need to have a way to match these new neuron names to the old ones
        # tracking_cfg = project_cfg.get_tracking_config()
        # fname = tracking_cfg.resolve_relative_path_from_config('global2tracklet_matches_fname')
        # old_global2tracklet = pickle_load_binary(fname)
        #
        # offset = 1
        # for i, old_matches in enumerate(old_global2tracklet.values()):
        #     new_neuron_name = int2name_neuron(i + offset)
        #     while new_neuron_name in global2tracklet_tmp:
        #         offset += 1
        #         new_neuron_name = int2name_neuron(i + offset)
        #     global2tracklet_tmp[new_neuron_name] = old_matches

    # Save and update configs
    training_cfg = project_cfg.get_training_config()
    out_fname = os.path.join('2-training_data', 'all_tracklets_with_ground_truth.pickle')
    training_cfg.pickle_in_local_project(df_final, relative_path=out_fname, custom_writer=pd.to_pickle)

    global2tracklet_matches_fname = os.path.join('3-tracking', 'global2tracklet_with_ground_truth.pickle')
    tracking_cfg.pickle_in_local_project(global2tracklet_new, global2tracklet_matches_fname)

    tracking_cfg.config['global2tracklet_matches_fname'] = global2tracklet_matches_fname
    training_cfg.config['df_3d_tracklets'] = out_fname
    training_cfg.update_on_disk()
    tracking_cfg.update_on_disk()

    return df_including_tracks, global2tracklet_new


def update_global2tracklet_dictionary(df_split, global2tracklet_original, name_mapping):
    logging.info("Updating the dictionary that matches the neurons and tracklets")
    # Start with the original matches
    global2tracklet_tmp = {}
    for neuron_name, single_match in global2tracklet_original.items():
        if single_match in name_mapping:
            global2tracklet_tmp[neuron_name] = list(name_mapping[single_match])
        else:
            global2tracklet_tmp[neuron_name] = [single_match]
    global2tracklet_new = remove_tracklets_from_dictionary_without_database_match(df_split, global2tracklet_tmp)
    return global2tracklet_new


def sanity_checks_on_dataframes(df_gt, df_tracklets):
    try:
        df_tracklets.drop(level=1, columns='raw_tracklet_id', inplace=True)
    except KeyError:
        pass
    check_if_heterogenous_columns(df_tracklets, raise_error=True)
    try:
        df_gt.drop(level=1, columns='raw_tracklet_id', inplace=True)
    except KeyError:
        pass
    check_if_heterogenous_columns(df_gt, raise_error=True)
