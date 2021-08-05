from DLC_for_WBFM.utils.preprocessing.utils_tif import preprocess_all_frames_using_config, PreprocessingSettings
from DLC_for_WBFM.utils.projects.utils_filepaths import resolve_mounted_path_in_current_os
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd, edit_config
import tifffile
import numpy as np
import os
from pathlib import Path
import zarr


def write_data_subset_from_config(cfg: dict,
                                  out_fname: str = None,
                                  vid_fname: str = None,
                                  tiff_not_zarr: bool = True,
                                  pad_to_align_with_original: bool = False,
                                  save_fname_in_red_not_green: bool = None,
                                  use_preprocessed_data: bool = False,
                                  preprocessing_settings: PreprocessingSettings = None,
                                  DEBUG: bool = False) -> None:
    """Takes the original giant .btf file from and writes the subset of the data as zarr or tiff"""

    out_fname, preprocessing_settings, project_dir, start_volume, verbose, vid_fname = _unpack_config_for_data_subset(
        cfg, out_fname, preprocessing_settings, save_fname_in_red_not_green, tiff_not_zarr, use_preprocessed_data,
        vid_fname)

    with safe_cd(project_dir):
        preprocessed_dat, _ = preprocess_all_frames_using_config(DEBUG, cfg, verbose, vid_fname,
                                                                 preprocessing_settings, None)

    if not pad_to_align_with_original:
        preprocessed_dat = preprocessed_dat[start_volume:, ...]

    if verbose >= 1:
        print(f"Writing array of size: {preprocessed_dat.shape}")

    if tiff_not_zarr:
        # Have to add a color channel to make format: TZCYX
        # Imagej seems to expect this weird format
        out_dat = np.expand_dims(preprocessed_dat, 2).astype('uint16')
        tifffile.imwrite(out_fname, out_dat, imagej=True, metadata={'axes': 'TZCYX'})
    else:
        chunk_sz = (1, ) + preprocessed_dat.shape[1:]
        print(f"Chunk size: {chunk_sz}")
        out_dat = np.array(preprocessed_dat)#.astype('uint16')
        zarr.save_array(out_fname, out_dat, chunks=chunk_sz)

    # Save this name in the config file itself
    if save_fname_in_red_not_green is not None:
        if save_fname_in_red_not_green:
            edits = {'preprocessed_red': out_fname}
        else:
            edits = {'preprocessed_green': out_fname}
        edit_config(cfg['project_path'], edits)


def _unpack_config_for_data_subset(cfg, out_fname, preprocessing_settings, save_fname_in_red_not_green, tiff_not_zarr,
                                   use_preprocessed_data, vid_fname):
    verbose = cfg['verbose']
    project_dir = cfg['project_dir']
    # preprocessing_fname = os.path.join('1-segmentation', 'preprocessing_config.yaml')
    if use_preprocessed_data:
        preprocessing_settings = None
        if verbose >= 1:
            print("Reusing already preprocessed data")
    elif preprocessing_settings is None:
        preprocessing_fname = cfg['preprocessing_config']
        preprocessing_settings = PreprocessingSettings.load_from_yaml(preprocessing_fname)
    if out_fname is None:
        if tiff_not_zarr:
            out_fname = os.path.join(project_dir, "data_subset.tiff")
        else:
            out_fname = os.path.join(project_dir, "data_subset.zarr")
    else:
        out_fname = os.path.join(project_dir, out_fname)
    if vid_fname is None:
        if save_fname_in_red_not_green:
            if not use_preprocessed_data:
                vid_fname = cfg['red_bigtiff_fname']
            else:
                vid_fname = cfg['preprocessed_red']
        else:
            if not use_preprocessed_data:
                vid_fname = cfg['green_bigtiff_fname']
            else:
                vid_fname = cfg['preprocessed_green']
        vid_fname = resolve_mounted_path_in_current_os(vid_fname)
    start_volume = cfg['dataset_params']['start_volume']
    return out_fname, preprocessing_settings, project_dir, start_volume, verbose, vid_fname


def segment_local_data_subset(project_config, out_fname=None):
    """
    Segments a dataset that has been copied locally; assumed to be named 'data_subset.tif'

    Applies NO preprocessing; assumes that is done with the subset already

    See also: write_data_subset_from_config
    """
    from segmentation.util.utils_pipeline import _segment_full_video_3d, _segment_full_video_2d

    cfg = load_config(project_config)
    project_dir = Path(project_config).parent

    with safe_cd(project_dir):
        segment_cfg = load_config(cfg['subfolder_configs']['segmentation'])

    if out_fname is None:
        out_fname = "masks_subset.zarr"
    mask_fname = os.path.join("1-segmentation", out_fname)
    metadata_fname = os.path.join("1-segmentation", "metadata_subset.pickle")

    video_path = "data_subset.tiff"
    verbose = cfg['verbose']
    num_slices = cfg['dataset_params']['num_slices']
    num_frames = cfg['dataset_params']['num_frames']
    preprocessing_settings = None
    frame_list = list(range(num_frames))
    metadata = {}

    model_type = segment_cfg['segmentation_type']
    if model_type=='3d':
        stardist_model_name = "charlie_3d"
        with safe_cd(project_dir):
            _segment_full_video_3d(cfg, frame_list, mask_fname, metadata, metadata_fname, num_frames, num_slices,
                                   preprocessing_settings, stardist_model_name, verbose, video_path)
    else:
        stardist_model_name = segment_cfg['segmentation_params']['stardist_model_name']
        opt_postprocessing = segment_cfg['postprocessing_params']
        with safe_cd(project_dir):
            _segment_full_video_2d(cfg, frame_list, mask_fname, metadata, metadata_fname, num_frames, num_slices,
                                   opt_postprocessing, preprocessing_settings, stardist_model_name, verbose, video_path)

