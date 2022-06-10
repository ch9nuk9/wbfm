import logging
import os

from DLC_for_WBFM.utils.neuron_matching.class_frame_pair import FramePairOptions
from DLC_for_WBFM.utils.neuron_matching.feature_pipeline import build_tracklets_full_video, \
    calculate_frame_objects_full_video
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.tracklets.tracklet_pipeline import _unpack_config_frame2frame_matches, _save_matches_and_frames, \
    build_frame_pairs_using_superglue, _unpack_config_for_tracklets, postprocess_matches_to_tracklets, \
    filter_tracklets_using_volume, save_all_tracklets
from DLC_for_WBFM.utils.tracklets.training_data_from_tracklets import convert_training_dataframe_to_scalar_format


def build_frames_and_adjacent_matches_using_config(project_config: ModularProjectConfig,
                                                   training_config: SubfolderConfigFile,
                                                   use_superglue: bool,
                                                   DEBUG: bool = False) -> None:
    """
    Produce training data via partial tracking using 3d feature-based method

    This function is designed to be used with an external .yaml config file

    See new_project_defaults/2-training_data/training_data_config.yaml
    See also track_neurons_full_video()
    """

    project_config.logger.info(f"Producing tracklets")

    project_data = ProjectData.load_final_project_data_from_config(project_config)
    video_data = project_data.red_data
    raw_fname = training_config.resolve_relative_path(os.path.join('raw', 'clust_df_dat.pickle'),
                                                      prepend_subfolder=True)
    if os.path.exists(raw_fname):
        raise FileExistsError(f"Found old raw data at {raw_fname}; either rename or skip this step to reuse")

    # Intermediate products: pairwise matches between frames
    video_fname, tracker_params, frame_pair_options = _unpack_config_frame2frame_matches(
        DEBUG, project_config, training_config)
    all_frame_pairs, all_frame_dict = build_tracklets_full_video(video_data, video_fname, **tracker_params,
                                                                 use_superglue=use_superglue,
                                                                 frame_pair_options=frame_pair_options)
    with safe_cd(project_config.project_dir):
        _save_matches_and_frames(all_frame_dict, all_frame_pairs, training_config)


def build_frame_pairs_using_superglue_using_config(project_cfg: ModularProjectConfig, DEBUG=False):
    project_data = ProjectData.load_final_project_data_from_config(project_cfg, to_load_frames=True)
    frame_pair_options = FramePairOptions.load_from_config_file(project_cfg)

    all_frame_dict = project_data.raw_frames
    assert all_frame_dict is not None, "Needs frame objects!"

    all_frame_pairs = build_frame_pairs_using_superglue(all_frame_dict, frame_pair_options, project_data)

    with safe_cd(project_cfg.project_dir):
        _save_matches_and_frames(all_frame_dict, all_frame_pairs, project_cfg.get_training_config())


def build_frame_objects_using_config(project_config: ModularProjectConfig,
                                     training_config: SubfolderConfigFile,
                                     only_calculate_desynced: bool = False,
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

    project_data = ProjectData.load_final_project_data_from_config(project_config,
                                                                   to_load_frames=only_calculate_desynced)
    video_data = project_data.red_data

    # Build frames, then match them
    if not only_calculate_desynced:
        end_volume = tracker_params['start_volume'] + tracker_params['num_frames']
        frame_range = list(range(tracker_params['start_volume'], end_volume))
        del tracker_params['num_frames']
        del tracker_params['start_volume']
    else:
        frame_range = project_data.get_desynced_seg_and_frame_object_frames()
        if len(frame_range) == 0:
            return
    all_new_frames = calculate_frame_objects_full_video(video_data, video_fname=video_fname, frame_range=frame_range,
                                                        **tracker_params)
    if not only_calculate_desynced:
        all_frame_dict = all_new_frames
    else:
        all_frame_dict = project_data.raw_frames
        all_frame_dict.update(all_new_frames)

    with safe_cd(project_config.project_dir):
        _save_matches_and_frames(all_frame_dict, None, training_config)


def postprocess_matches_to_tracklets_using_config(project_config: ModularProjectConfig,
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
    df_custom_format = postprocess_matches_to_tracklets(all_frame_dict, all_frame_pairs,
                                                        z_threshold, min_confidence,
                                                        logger=project_config.logger)
    # Overwrite intermediate products, because the pair objects save the postprocessing options
    with safe_cd(training_config.project_dir):
        _save_matches_and_frames(all_frame_dict, all_frame_pairs, training_config)

    # Convert to easier format and save
    min_length = postprocessing_params['min_length_to_save']
    df_multi_index_format = convert_training_dataframe_to_scalar_format(df_custom_format,
                                                                        min_length=min_length,
                                                                        scorer=None,
                                                                        segmentation_metadata=segmentation_metadata,
                                                                        logger=project_config.logger)
    volume_threshold = postprocessing_params.get('volume_percent_threshold', 0)
    if volume_threshold > 0:
        logging.info(f"Postprocessing using volume threshold: {volume_threshold}")
        df_multi_index_format, split_times = filter_tracklets_using_volume(df_multi_index_format,
                                                                           **postprocessing_params)
        out_fname = '2-training_data/raw/volume_tracklet_split_points.pickle'
        training_config.pickle_data_in_local_project(split_times, out_fname)

    save_all_tracklets(df_custom_format, df_multi_index_format, training_config)
