import logging
import pickle

from wbfm.utils.projects.utils_filenames import pickle_load_binary

from segmentation.util.utils_paths import get_output_fnames


def _unpack_config_file(preprocessing_cfg, segment_cfg, project_cfg, DEBUG):
    # Initializing variables
    start_volume = project_cfg.config['dataset_params']['start_volume']
    num_frames = project_cfg.config['dataset_params']['num_frames']
    if DEBUG:
        num_frames = 1
    frame_list = list(range(start_volume, start_volume + num_frames))
    video_path = project_cfg.config['preprocessed_red']
    # Generate new filenames if they are not set
    mask_fname = segment_cfg.config['output_masks']
    metadata_fname = segment_cfg.config['output_metadata']
    output_dir = segment_cfg.config['output_folder']
    mask_fname, metadata_fname = get_output_fnames(video_path, output_dir, mask_fname, metadata_fname)
    metadata_fname = segment_cfg.unresolve_absolute_path(metadata_fname)
    mask_fname = segment_cfg.unresolve_absolute_path(mask_fname)
    # Save settings
    segment_cfg.config['output_masks'] = mask_fname
    segment_cfg.config['output_metadata'] = metadata_fname
    verbose = project_cfg.config['verbose']
    stardist_model_name = segment_cfg.config['segmentation_params']['stardist_model_name']
    zero_out_borders = segment_cfg.config['segmentation_params']['zero_out_borders']
    # Preprocessing information
    bbox_fname = preprocessing_cfg.config.get('bounding_boxes_fname', None)
    if bbox_fname is not None:
        all_bounding_boxes = pickle_load_binary(bbox_fname)
        project_cfg.logger.info(f"Found bounding boxes at: {bbox_fname}")
    else:
        all_bounding_boxes = None
        project_cfg.logger.warning(f"Did not find bounding boxes at: {bbox_fname},"
                                   f"a large number of false positive segmentations might be generated.")
    sum_red_and_green_channels = segment_cfg.config['segmentation_params'].get('sum_red_and_green_channels', False)
    if sum_red_and_green_channels:
        project_cfg.logger.warning("Summing red and green channels for segmentation; does not affect metadata.")
    return (frame_list, mask_fname, metadata_fname, num_frames, stardist_model_name, verbose, video_path,
            zero_out_borders, all_bounding_boxes, sum_red_and_green_channels)
