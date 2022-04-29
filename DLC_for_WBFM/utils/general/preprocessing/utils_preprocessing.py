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
    out_fname_red_7z = zip_raw_data_zarr(project_cfg.config['preprocessed_red'])
    out_fname_green_7z = zip_raw_data_zarr(project_cfg.config['preprocessed_green'])

    project_cfg.config['preprocessed_red'] = out_fname_red_7z
    project_cfg.config['preprocessed_green'] = out_fname_green_7z
    project_cfg.update_self_on_disk()


def subtract_background_after_preprocessing_using_config(cfg: ModularProjectConfig, new_background=14):
    """Read a video of the background and the otherwise fully preprocessed data, and simply subtract"""

    preprocessing_settings = cfg.get_preprocessing_config()
    num_slices = cfg.config['dataset_params']['num_slices']

    red_fname = cfg.config['red_preprocessed_fname']
    red_data = zarr_reader_folder_or_zipstore(red_fname)

    red_background_fname = cfg.config['red_background_fname']
    num_frames = 71

    background_video_list = []
    with tifffile.TiffFile(red_background_fname) as background_tiff:
        for i in tqdm(range(num_frames)):
            background_volume = get_single_volume(background_tiff, i, num_slices, dtype='uint16')
            # Note: this will do rigid rotation
            background_volume = perform_preprocessing(background_volume, preprocessing_settings, i)
            background_video_list.append(background_volume)

    # Add a new truly constant background value, to keep anything from going negative
    red_background_video_mean = np.mean(background_video_list) + new_background

    # Don't try to modify the data as read; it is read-only
    # ... but this forces a full-memory copy of the data
    red_background_subtracted = zarr.zeros_like(red_data)
    red_background_subtracted[:] = red_data - red_background_video_mean

    # Save as normal folder, then zip using command line tool
    red_fname_subtracted = add_name_suffix(red_fname, '_background_subtracted')
    zarr.save_array(red_background_subtracted, red_fname_subtracted)
    final_fname_red_7z = zip_raw_data_zarr(red_fname_subtracted)

    cfg.config['preprocessed_red'] = final_fname_red_7z
    cfg.update_self_on_disk()
