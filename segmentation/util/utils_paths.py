import os.path as osp
from datetime import datetime
from pathlib import Path


def get_output_fnames(video_path, _config):
    if _config['output_params']['output_folder'] is None:
        output_folder = osp.split(video_path)[0]
        # Make a subfolder
        subfolder = datetime.now().strftime("%Y_%m_%d-%I_%M_%p")
        output_folder = osp.join(output_folder, subfolder)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    else:
        output_folder = _config['output_params']['output_folder']

    # Make a suffix
    num_frames = _config['dataset_params']['num_frames']

    # Actual masks
    fname = f'masks_{num_frames}.btf'
    mask_fname = osp.join(output_folder, fname)

    # Metadata
    fname = f'metadata_{num_frames}.pickle'
    metadata_fname = osp.join(output_folder, fname)

    return mask_fname, metadata_fname
