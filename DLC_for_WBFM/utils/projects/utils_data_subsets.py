from DLC_for_WBFM.utils.pipeline.dlc_pipeline import _preprocess_all_frames
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
import tifffile
import numpy as np
import os
from pathlib import Path
import zarr
from segmentation.util.utils_pipeline import _segment_full_video_3d


def write_data_subset_from_config(project_config, out_fname=None, tiff_not_zarr=True, pad_to_align_with_original=False):
    """Takes the original giant .btf file from and writes the subset of the data"""
    cfg = load_config(project_config)
    project_dir = Path(project_config).parent
    fname = os.path.join('1-segmentation', 'preprocessing_config.yaml')
    cfg['preprocessing_config'] = fname

    if out_fname is None:
        if tiff_not_zarr:
            out_fname = os.path.join(project_dir, "data_subset.tiff")
        else:
            out_fname = os.path.join(project_dir, "data_subset.zarr")
    vid_fname = cfg['red_bigtiff_fname']
    verbose = cfg['other']['verbose']
    DEBUG = False

    with safe_cd(project_dir):
        preprocessed_dat, _ = _preprocess_all_frames(DEBUG, cfg, verbose, vid_fname, None)

    if not pad_to_align_with_original:
        start_volume = cfg['dataset_params']['start_volume']
        preprocessed_dat = preprocessed_dat[start_volume:, ...]

    print(f"Writing array of size: {preprocessed_dat.shape}")

    if tiff_not_zarr:
        # Have to add a color channel to make format: TZCYX
        # Imagej seems to expect this weird format
        out_dat = np.expand_dims(preprocessed_dat, 2).astype('uint16')
        tifffile.imwrite(out_fname, out_dat, imagej=True, metadata={'axes': 'TZCYX'})
    else:
        chunk_sz = (1, ) + preprocessed_dat.shape[1:]
        print(f"Chunk size: {chunk_sz}")
        out_dat = np.array(preprocessed_dat).astype('uint16')
        zarr.save_array(out_fname, out_dat, chunks=chunk_sz)


def segment_local_data_subset(project_config, max_frames=100, out_fname="masks_subset.zarr"):
    """
    Segments a dataset that has been copied locally; assumed to be named 'data_subset.zarr'

    Applies NO preprocessing; assumes that is done with the subset already

    See also: write_data_subset_from_config
    """
    cfg = load_config(project_config)
    project_dir = Path(project_config).parent

    mask_fname = os.path.join("1-segmentation", out_fname)
    metadata_fname = os.path.join("1-segmentation", "metadata_subset.pickle")

    video_path = "data_subset.zarr"
    verbose = cfg['other']['verbose']
    stardist_model_name = "charlie_3d"
    num_slices = cfg['dataset_params']['num_slices']
    num_frames = cfg['dataset_params']['num_frames']
    preprocessing_settings = None
    frame_list = list(range(num_frames))
    metadata = {}

    with safe_cd(project_dir):
        _segment_full_video_3d(cfg, frame_list, mask_fname, metadata, metadata_fname, num_frames, num_slices,
                               preprocessing_settings, stardist_model_name, verbose, video_path)
