import os.path as osp
from datetime import datetime
from pathlib import Path


def get_output_fnames(video_path, num_frames, output_folder):
    if output_folder is None:
        output_folder = osp.split(video_path)[0]
        subfolder = datetime.now().strftime("%Y_%m_%d-%I_%M_%p")
        output_folder = osp.join(output_folder, subfolder)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    else:
        output_folder = output_folder

    # Actual masks
    fname = f'masks_{num_frames}.zarr'
    mask_fname = osp.join(output_folder, fname)

    # Metadata
    fname = f'metadata_{num_frames}.pickle'
    metadata_fname = osp.join(output_folder, fname)

    return mask_fname, metadata_fname
