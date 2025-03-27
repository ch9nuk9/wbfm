import os
import shutil
from pathlib import Path

import numpy as np
import zarr
from imutils import MicroscopeDataReader
import dask.array as da
from skimage.transform import resize
from wbfm.utils.external.custom_errors import NoNeuropalError
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.segmentation.util.utils_metadata import calc_metadata_full_video
from wbfm.utils.segmentation.util.utils_model import segment_with_stardist_3d, get_stardist_model


def add_neuropal_to_project(project_path, neuropal_path, copy_data=True):
    """
    Adds a neuropal dataset to an existing project, without analyzing it.

    Parameters
    ----------
    project
    neuropal_path

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data(project_path)
    neuropal_config = project_data.project_config.initialize_neuropal_subproject()
    target_dir = neuropal_config.absolute_subfolder

    # Make sure we have the path to the folder, not just the file
    if os.path.isfile(neuropal_path):
        neuropal_dir = os.path.dirname(neuropal_path)
    else:
        neuropal_dir = neuropal_path

    # Check: make sure this is readable by Lukas' reader
    # Note that we need the ome.tif file within the neuropal directory
    for file in Path(neuropal_dir).iterdir():
        if file.is_dir():
            continue
        elif str(file).endswith('ome.tif'):
            neuropal_path = str(file)
            break
    else:
        raise FileNotFoundError(f'Could not find the ome.tif file in {neuropal_dir}')
    _ = MicroscopeDataReader(neuropal_path)

    # Move or copy all contents from the neuropal folder project directory
    if copy_data:
        shutil.copytree(neuropal_dir, target_dir, dirs_exist_ok=True)
    else:
        shutil.move(neuropal_dir, target_dir)

    # Update the config file with the new data path
    neuropal_data_path = os.path.join(target_dir, os.path.basename(neuropal_path))
    neuropal_config.config['neuropal_data_path'] = neuropal_config.unresolve_absolute_path(neuropal_data_path)
    neuropal_config.update_self_on_disk()


def segment_neuropal_from_project(project_data, subsample_in_z=True):
    """
    Segments the neuropal dataset in a project.

    Note that the video is by default much higher resolution in z than the fluorescence data, so we subsample in z
    to make the neural network work

    Parameters
    ----------
    project_data : ProjectData

    Returns
    -------

    """
    try:
        neuropal_config = project_data.project_config.get_neuropal_config()
    except FileNotFoundError:
        raise NoNeuropalError(project_data.project_dir)

    # Get raw data
    neuropal_path = neuropal_config.resolve_relative_path_from_config('neuropal_data_path')
    neuropal_data = MicroscopeDataReader(neuropal_path)

    # Sum channels to get volume that will actually be segmented
    channels_to_sum = neuropal_config.config['segmentation_params']['channels_to_sum']
    multichannel_volume = da.squeeze(neuropal_data.dask_array)
    summed_volume = multichannel_volume[channels_to_sum].sum(axis=0).compute()

    # Preprocess volume to make it more similar to the fluorescence data
    if subsample_in_z:
        # Get ratio between z resolutions
        z_np = project_data.physical_unit_conversion.zimmer_um_per_pixel_z_neuropal
        z_fluo = project_data.physical_unit_conversion.zimmer_um_per_pixel_z
        z_zoom = z_np / z_fluo
        target_shape = (int(summed_volume.shape[0] * z_zoom), summed_volume.shape[1], summed_volume.shape[2])
        resized_volume = resize(summed_volume, target_shape, order=3)
        project_data.logger.info(f"Subsampling in z to shape {resized_volume.shape} from {multichannel_volume.shape}")

    # Get segmentation model
    stardist_model_name = neuropal_config.config['segmentation_params']['stardist_model_name']
    if stardist_model_name is None:
        segmentation_config = project_data.project_config.get_segmentation_config()
        stardist_model_name = segmentation_config.config['segmentation_params']['stardist_model_name']
    sd_model = get_stardist_model(stardist_model_name)

    # Segment
    output_fname = neuropal_config.config['neuropal_segmentation_path']
    if output_fname is None:
        output_fname = os.path.join('neuropal', 'neuropal_masks.zarr')
    output_fname = neuropal_config.resolve_relative_path(output_fname)

    if subsample_in_z:
        final_masks = segment_with_stardist_3d(resized_volume, sd_model)
        # Expand the masks back to the original z resolution
        target_shape = summed_volume.shape
        final_masks = resize(final_masks, target_shape, order=0)
    else:
        final_masks = segment_with_stardist_3d(summed_volume, sd_model)
    final_masks = np.array(final_masks, dtype=np.uint16)

    # Calculate metadata for the segmentation (same as main segmentation)
    metadata_fname = neuropal_config.config['segmentation_metadata_path']
    if metadata_fname is None:
        metadata_fname = os.path.join('neuropal', 'neuropal_metadata.pickle')
    metadata_fname = neuropal_config.resolve_relative_path(metadata_fname)
    # Expand the time dimension
    frame_list = [0]
    calc_metadata_full_video(frame_list, np.expand_dims(final_masks, axis=0),
                             np.expand_dims(summed_volume, axis=0), metadata_fname)

    # Save the segmentation and filenames
    project_data.logger.info(f"Saving segmentation to {output_fname} with shape {final_masks.shape}")
    sz = final_masks.shape
    chunks = sz
    masks_zarr = zarr.open(output_fname, mode='w', shape=sz, chunks=chunks, dtype=np.uint16, fill_value=0)
    masks_zarr[:] = final_masks[:]

    # Update the config files with the new data path
    neuropal_config.config['neuropal_segmentation_path'] = neuropal_config.unresolve_absolute_path(output_fname)
    neuropal_config.config['segmentation_metadata_path'] = neuropal_config.unresolve_absolute_path(metadata_fname)
    neuropal_config.update_self_on_disk()

