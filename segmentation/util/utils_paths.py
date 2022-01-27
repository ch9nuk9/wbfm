import os.path as osp
from datetime import datetime
from pathlib import Path


def get_output_fnames(video_path, output_folder, mask_fname, metadata_fname):
    if output_folder is None:
        output_folder = osp.split(video_path)[0]
        subfolder = datetime.now().strftime("%Y_%m_%d-%I_%M_%p")
        output_folder = osp.join(output_folder, subfolder)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    else:
        output_folder = output_folder

    # Actual masks
    if mask_fname is None:
        fname = 'masks.zarr'
        mask_fname = osp.join(output_folder, fname)

    # Metadata
    if metadata_fname is None:
        fname = 'metadata.pickle'
        metadata_fname = osp.join(output_folder, fname)

    return mask_fname, metadata_fname
