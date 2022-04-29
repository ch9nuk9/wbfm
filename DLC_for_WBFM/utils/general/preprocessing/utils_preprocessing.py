import concurrent.futures
import logging

import numpy as np
import zarr
from tifffile import tifffile
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_zarr import zip_raw_data_zarr, zarr_reader_folder_or_zipstore
from DLC_for_WBFM.utils.general.preprocessing.utils_tif import perform_preprocessing
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_filenames import add_name_suffix
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume


def zip_zarr_using_config(project_cfg: ModularProjectConfig):
    logging.info("Zipping zarr data (both channels)")
    out_fname_red_7z = zip_raw_data_zarr(project_cfg.config['preprocessed_red'], verbose=1)
    out_fname_green_7z = zip_raw_data_zarr(project_cfg.config['preprocessed_green'], verbose=1)

    project_cfg.config['preprocessed_red'] = str(out_fname_red_7z)
    project_cfg.config['preprocessed_green'] = str(out_fname_green_7z)
    project_cfg.update_self_on_disk()


def subtract_background_after_preprocessing_using_config(cfg: ModularProjectConfig, DEBUG=False):
    """Read a video of the background and the otherwise fully preprocessed data, and simply subtract"""

    preprocessing_settings = cfg.get_preprocessing_config()
    num_slices = cfg.config['dataset_params']['num_slices']
    num_frames = 50  # TODO: is this constant?
    if DEBUG:
        num_frames = 2

    opt = dict(num_frames=num_frames, num_slices=num_slices, preprocessing_settings=preprocessing_settings)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        red_fname_subtracted = ex.submit(background_subtract_single_channel, cfg, 'red', **opt).result()
        green_fname_subtracted = ex.submit(background_subtract_single_channel, cfg, 'green', **opt).result()
    cfg.config['preprocessed_red'] = str(red_fname_subtracted)
    cfg.config['preprocessed_green'] = str(green_fname_subtracted)

    zip_zarr_using_config(cfg)


def background_subtract_single_channel(cfg, which_channel, num_frames, num_slices, preprocessing_settings):
    raw_fname = cfg.config[f'preprocessed_{which_channel}']
    raw_data = zarr_reader_folder_or_zipstore(raw_fname)
    background_video_list = read_background_from_config(cfg, num_frames, num_slices, preprocessing_settings,
                                                        which_channel)
    # Add a new truly constant background value, to keep anything from going negative
    new_background = preprocessing_settings.background_default_after_subtraction
    background_video_mean = np.mean(background_video_list) + new_background
    # Don't try to modify the data as read; it is read-only
    # ... but this forces a full-memory copy of the data
    fname_subtracted = add_name_suffix(raw_fname, '_background_subtracted')
    logging.info(f"Creating data copy at {fname_subtracted}")
    store = zarr.DirectoryStore(fname_subtracted)
    background_subtracted = zarr.zeros_like(raw_data, store=store)
    # Loop so that not all is loaded in memory... should I use dask?
    for i, volume in enumerate(tqdm(raw_data)):
        background_subtracted[i, ...] = volume - background_video_mean
    # zarr.save_array(background_subtracted, fname_subtracted)

    return fname_subtracted


def read_background_from_config(cfg, num_frames, num_slices, preprocessing_settings, which_channel):
    background_fname = cfg.config[f'{which_channel}_background_fname']
    background_video_list = []
    with tifffile.TiffFile(background_fname) as background_tiff:
        for i in tqdm(range(num_frames)):
            background_volume = get_single_volume(background_tiff, i, num_slices, dtype='uint16')
            # Note: this will do rigid rotation
            background_volume = perform_preprocessing(background_volume, preprocessing_settings, i)
            background_video_list.append(background_volume)
    logging.info(f"Read background tiff file of shape: {background_video_list[0].shape}")
    return background_video_list

