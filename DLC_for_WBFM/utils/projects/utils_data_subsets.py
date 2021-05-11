from DLC_for_WBFM.utils.pipeline.dlc_pipeline import _preprocess_all_frames
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
import tifffile
import numpy as np
import os
from pathlib import Path


def write_data_subset_from_config(project_config, out_fname=None):
    """Takes the original giant .btf file from and writes the subset of the data"""
    cfg = load_config(project_config)
    project_dir = Path(project_config).parent
    fname = os.path.join('1-segmentation', 'preprocessing_config.yaml')
    cfg['preprocessing_config'] = fname

    if out_fname is None:
        out_fname = os.path.join(project_dir, "data_subset.tiff")
    vid_fname = cfg['red_bigtiff_fname']
    verbose = cfg['other']['verbose']
    DEBUG = False

    with safe_cd(project_dir):
        preprocessed_dat, _ = _preprocess_all_frames(DEBUG, cfg, verbose, vid_fname, [])

    # Have to add a color channel to make format: TZCYX
    out_dat = np.expand_dims(preprocessed_dat, 2).astype('uint16')
    tifffile.imwrite(out_fname, out_dat, imagej=True, metadata={'axes': 'TZCYX'})