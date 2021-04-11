import os.path as osp
from datetime import datetime

def get_output_fnames(video_path, _config):
    if _config['output_params']['output_folder'] is None:
        output_folder = osp.split(video_path)[0]
    else:
        output_folder = _config['output_params']['output_folder']

    suffix = datetime.now().strftime("%Y_%m_%d-%I_%M_%p")
    num_frames = _config['dataset_params']['num_frames']

    # Actual masks
    fname = f'masks_{num_frames}_{suffix}.btf'
    mask_fname = osp.join(output_folder, fname)

    # Metadata
    fname = f'metadata_{num_frames}_{suffix}.btf'
    metadata_fname = osp.join(output_folder, fname)

    return mask_fname, metadata_fname
