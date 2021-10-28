"""
The metadata generator for the segmentation pipeline
"""
import concurrent.futures
import logging
import pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd
import zarr
from skimage.measure import label, regionprops
from tqdm import tqdm

from segmentation.util.utils_config_files import _unpack_config_file


def get_metadata_dictionary(masks, original_vol):
    """
    Creates a dataframe with metadata ('total_brightness', 'neuron_volume', 'centroids') for a volume
    Parameters
    ----------
    masks : 3D numpy array
        contains the segmented and stitched masks
    original_vol : 3d numpy array
        original volume of the recording, which was segmented into 'masks'.
        Contains the actual brightness values.
    planes = : dict(list)
        Dictionary containing a list of Z-slices on which a neuron is present
        neuron_ID = 1
        planes[neuron_ID] # [12, 13, 14]

    Returns
    -------
    df : pandas dataframe
        for each neuron (=row) the total brightness, volume and centroid is saved
        dataframe(columns = 'total_brightness', 'neuron_volume', 'centroids';
                  rows = neuron #)
    """
    # metadata_dict = {(Vol #, Neuron #) = [Total brightness, neuron volume, centroids]}
    neurons_list = np.unique(masks)
    neurons_list = np.delete(neurons_list, np.where(neurons_list == 0))
    neurons = []
    # create list of integers for each neuron ID (instead of float)
    for x in neurons_list:
        neurons.append(int(x))

    # create dataframe with
    # cols = (total brightness, volume, centroids, all_values (full histogram))
    # rows = neuron ID
    df = pd.DataFrame(index=neurons,
                      columns=['total_brightness', 'neuron_volume', 'centroids', 'all_values'])

    for n in neurons:
        neuron_mask = masks == n
        neuron_vol = np.count_nonzero(neuron_mask)

        original_vals = original_vol[neuron_mask]
        total_brightness = np.sum(original_vals)
        vals = np.ravel(original_vals)
        # vals, counts = np.unique(original_vals, return_counts=True)

        neuron_label = label(neuron_mask)
        # NOTE: for skimage>0.19 this property changes name
        try:
            centroids = regionprops(neuron_label, intensity_image=original_vol)[0].weighted_centroid
        except AttributeError:
            centroids = regionprops(neuron_label, intensity_image=original_vol)[0].centroid_weighted

        df.at[n, 'total_brightness'] = total_brightness
        df.at[n, 'neuron_volume'] = neuron_vol
        df.at[n, 'centroids'] = centroids
        df.at[n, 'all_values'] = vals
        # df.at[n, 'pixel_counts'] = counts

    return df


def centroids_from_dict_of_dataframes(dict_of_dataframes, i_volume) -> np.ndarray:
    vol0_zxy = dict_of_dataframes[i_volume]['centroids'].to_numpy()
    return np.array([np.array(m) for m in vol0_zxy])


@dataclass
class DetectedNeurons:

    detection_fname: str

    _segmentation_metadata: dict = None
    _num_frames: int = None

    @property
    def segmentation_metadata(self):
        if self._segmentation_metadata is None:

            with open(self.detection_fname, 'rb') as f:
                # Note: dict of dataframes
                self._segmentation_metadata = pickle.load(f)
        return self._segmentation_metadata

    @property
    def num_frames(self):
        if self._num_frames is None:
            self._num_frames = len(self.segmentation_metadata)
        return self._num_frames

    def detect_neurons_from_file(self, i_volume: int, numpy_not_list=True) -> np.ndarray:
        """
        Designed to be used with centroids detected using a different pipeline
        """
        if numpy_not_list:
            neuron_locs = centroids_from_dict_of_dataframes(self.segmentation_metadata, i_volume)
        else:
            neuron_locs = self.segmentation_metadata[i_volume]['centroids']
            neuron_locs = np.array([n for n in neuron_locs])

        if len(neuron_locs) > 0:
            pass
            # neuron_locs = neuron_locs[:, [0, 2, 1]]
        else:
            neuron_locs = []

        return neuron_locs


def recalculate_metadata_from_config(segment_cfg, project_cfg, DEBUG=False):
    """

    Given a project that contains a segmentation, recalculate the metadata

    Parameters
    ----------
    _config : dict, loaded from project yaml file

    Returns
    -------
    Saves metadata.pickle to disk (within folder 1-segmentation)

    See also:
        segment_video_using_config_3d

    """

    frame_list, mask_fname, metadata_fname, _, _, _, video_path, _, _ = _unpack_config_file(
        segment_cfg, project_cfg, DEBUG)

    masks_zarr = zarr.open(mask_fname, synchronizer=zarr.ThreadSynchronizer())
    video_dat = zarr.open(video_path, synchronizer=zarr.ThreadSynchronizer())
    logging.info(f"Read zarr from: {mask_fname} with size {masks_zarr.shape}")
    logging.info(f"Read video from: {video_path} with size {video_dat.shape}")

    if DEBUG:
        frame_list = frame_list[:2]

    calc_metadata_full_video(frame_list, masks_zarr, video_dat, metadata_fname)


def calc_metadata_full_video(frame_list: list, masks_zarr: zarr.Array, video_dat: zarr.Array,
                             metadata_fname: str) -> None:
    """
    Calculates metadata once segmentation is finished

    Parameters
    ----------
    frame_list
    masks_zarr
    video_dat
    metadata_fname
    """
    metadata = dict()

    # Loop again in order to calculate metadata and possibly postprocess
    # with tifffile.TiffFile(video_path) as video_stream:
    #     for i_rel, i_abs in tqdm(enumerate(frame_list), total=len(frame_list)):
    #         masks = masks_zarr[i_rel, :, :, :]
    #         # TODO: Use a disk-saved preprocessing artifact instead of recalculating
    #         volume = _get_and_prepare_volume(i_abs, num_slices, preprocessing_settings, video_path=video_stream)
    #
    #         metadata[i_abs] = get_metadata_dictionary(masks, volume)

    with tqdm(total=len(frame_list)) as pbar:
        def parallel_func(i_both):
            i_out, i_vol = i_both
            masks = masks_zarr[i_out, :, :, :]
            volume = video_dat[i_vol, ...]
            metadata[i_vol] = get_metadata_dictionary(masks, volume)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(parallel_func, i): i for i in enumerate(frame_list)}
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    # saving metadata and settings
    with open(metadata_fname, 'wb') as meta_save:
        pickle.dump(metadata, meta_save)