import os
import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import zarr
from imutils import MicroscopeDataReader
import dask.array as da
from skimage.transform import resize
from wbfm.utils.external.custom_errors import NoNeuropalError, IncompleteConfigFileError
from wbfm.utils.general.utils_filenames import load_file_according_to_precedence, read_if_exists
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import SubfolderConfigFile
from wbfm.utils.segmentation.util.utils_metadata import calc_metadata_full_video, DetectedNeurons
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
    # Expand the time dimension, and
    frame_list = [0]
    volume = np.expand_dims(np.transpose(multichannel_volume.compute(), (1, 2, 3, 0)), axis=0)
    calc_metadata_full_video(frame_list, np.expand_dims(final_masks, axis=0), volume, metadata_fname)

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


@dataclass
class NeuropalManager:

    config: SubfolderConfigFile

    df_id_fname: str = None

    @cached_property
    def data(self) -> Optional[da.Array]:
        """
        Data from the neuropal dataset, if present
        """
        if self.config is None:
            return None
        # Should always exist if the config exists
        neuropal_path = self.config.resolve_relative_path_from_config('neuropal_data_path')
        neuropal_data = MicroscopeDataReader(neuropal_path)
        return da.squeeze(neuropal_data.dask_array)

    @cached_property
    def segmentation(self) -> Optional[da.Array]:
        """
        Segmentation from the neuropal dataset, if present
        """
        if self.config is None:
            return None
        # May not exist if the segmentation step has not been run
        neuropal_path = self.config.resolve_relative_path_from_config('neuropal_segmentation_path')
        if neuropal_path is None:
            return None
        neuropal_segmentation = zarr.open(neuropal_path)
        return neuropal_segmentation

    @cached_property
    def segmentation_metadata(self) -> Optional[DetectedNeurons]:
        """
        Metadata from the neuropal segmentation

        Returns
        -------

        """
        if self.config is None:
            return None
        # May not exist if the segmentation step has not been run
        neuropal_path = self.config.resolve_relative_path_from_config('segmentation_metadata_path')
        if neuropal_path is None:
            return None
        neuropal_segmentation_metadata = DetectedNeurons(neuropal_path)
        return neuropal_segmentation_metadata

    @property
    def has_complete_neuropal(self):
        return self.data is not None and self.segmentation is not None

    @property
    def neuron_names(self) -> List[str]:
        """Get the names from the segmentation metadata"""
        df = self.segmentation_metadata.get_all_neuron_metadata_for_single_time(0, as_dataframe=True)
        # Get the top level of the multiindex
        return df.index.get_level_values(0).unique().tolist()

    @cached_property
    def df_ids(self) -> Optional[pd.DataFrame]:
        """
        Load a dataframe corresponding to manual iding based on the neuropal stacks, if they exist

        This will not exist for every project

        """
        if self.config is None:
            return None
        # Manual annotations take precedence by default
        excel_fname = self.get_default_manual_annotation_fname()
        try:
            possible_fnames = dict(excel=excel_fname)
        except ValueError:
            self.df_id_fname = ''
            return None
        possible_fnames = {k: str(v) for k, v in possible_fnames.items()}
        fname_precedence = ['newest']
        try:
            df_neuropal_id, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                      reader_func=read_if_exists,
                                                                      na_filter=False)
        except ValueError:
            # Then the file was corrupted... try to load from the h5 file (don't worry about other file types)
            self.logger.warning(f"Found corrupted neuropal annotation file ({excel_fname}), "
                                f"with no backup h5 file; returning None")
            fname = ''
            df_neuropal_id = None
        self.df_id_fname = fname
        return df_neuropal_id

    def get_default_manual_annotation_fname(self) -> Optional[str]:
        if self.config is None:
            raise IncompleteConfigFileError("No project config found; cannot load or save manual annotations")
        excel_fname = self.config.resolve_relative_path("manual_annotation.xlsx", prepend_subfolder=True)
        return excel_fname
