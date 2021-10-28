import pickle

from segmentation.util.utils_paths import get_output_fnames


def _unpack_config_file(segment_cfg, project_cfg, DEBUG):
    # Initializing variables
    start_volume = project_cfg.config['dataset_params']['start_volume']
    num_frames = project_cfg.config['dataset_params']['num_frames']
    if DEBUG:
        num_frames = 1
    frame_list = list(range(start_volume, start_volume + num_frames))
    video_path = segment_cfg.config['video_path']
    # Generate new filenames if they are not set
    mask_fname = segment_cfg.config['output_masks']
    metadata_fname = segment_cfg.config['output_metadata']
    output_dir = segment_cfg.config['output_folder']
    mask_fname, metadata_fname = get_output_fnames(video_path, output_dir, mask_fname, metadata_fname)
    # Save settings
    segment_cfg.config['output_masks'] = mask_fname
    segment_cfg.config['output_metadata'] = metadata_fname
    verbose = project_cfg.config['verbose']
    stardist_model_name = segment_cfg.config['segmentation_params']['stardist_model_name']
    zero_out_borders = segment_cfg.config['segmentation_params']['zero_out_borders']
    # Preprocessing information
    bbox_fname = segment_cfg.config.get('bbox_fname', None)
    if bbox_fname is not None:
        with open(bbox_fname, 'rb') as f:
            all_bounding_boxes = pickle.load(f)
    else:
        all_bounding_boxes = None
    return frame_list, mask_fname, metadata_fname, num_frames, stardist_model_name, verbose, video_path, zero_out_borders, all_bounding_boxes