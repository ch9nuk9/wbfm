import segmentation.util.utils_postprocessing as post
import numpy as np
from tqdm import tqdm
import pickle
import os
import tifffile as tiff
# preprocessing
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
from DLC_for_WBFM.utils.preprocessing.utils_tif import perform_preprocessing
from DLC_for_WBFM.utils.projects.utils_project import edit_config
# metadata
from segmentation.util.utils_metadata import get_metadata_dictionary
from segmentation.util.utils_paths import get_output_fnames
import zarr


def segment_video_using_config_2d(_config):
    """
    Full pipeline based on only a config file

    See segment2d.py for parameter documentation
    """

    # Initializing variables
    start_volume = _config['dataset_params']['start_volume']
    num_frames = _config['dataset_params']['num_frames']
    num_slices = _config['dataset_params']['num_slices']
    video_path = _config['video_path']
    mask_fname, metadata_fname = get_output_fnames(video_path, _config)
    # Initialize zarr output file (format: TZXY)
    # vol = get_single_volume(video_path, 0, num_slices)
    # _, x_sz, y_sz = vol.shape
    # Save settings
    _config['output']['masks'] = mask_fname
    _config['output']['metadata'] = metadata_fname

    verbose = _config['verbose']
    metadata = dict()
    preprocessing_settings = PreprocessingSettings.load_from_yaml(
        _config['preprocessing_config']
    )
    if verbose >= 1:
        print("Loaded preprocessing_settings:")
        print(preprocessing_settings)

    # get stardist model object
    from segmentation.util.utils_model import get_stardist_model
    from segmentation.util.utils_model import segment_with_stardist_2d
    stardist_model_name = _config['segmentation_params']['stardist_model_name']
    sd_model = get_stardist_model(stardist_model_name, verbose=verbose-1)

    if verbose >= 1:
        print(f"Starting loop over {num_frames} frames")
    for i in tqdm(list(range(start_volume, start_volume + num_frames))):
        # use get single volume function from charlie
        import_opt = {'which_vol': i, 'num_slices': num_slices, 'alpha': 1.0, 'dtype': 'uint16'}
        volume = get_single_volume(video_path, **import_opt)

        if _config['DEBUG']:
            break

        # preprocess
        if verbose >= 2:
            print(f'--- Volume {i}/{num_frames} ---')
        volume = perform_preprocessing(volume, preprocessing_settings)

        # segment the volume using Stardist
        if verbose >= 2:
            print('--- Segmentation ---')
        segmented_masks = segment_with_stardist_2d(volume, sd_model, verbose=verbose-1)

        # process masks: remove large areas, stitch, split long neurons, remove short neurons
        if verbose >= 2:
            print('---- Post-processing ----')
        opt = _config['postprocessing_params']
        final_masks = perform_post_processing_2d(segmented_masks,
                                                 volume,
                                                 **opt,
                                                 verbose=verbose-1)

        if verbose >= 2:
            print(f"Found {len(np.unique(final_masks))} masks")
            print('----- Saving to BIG-TIF -----')

        # Add masks to zarr file; automatically saves
        if i == start_volume:
            _, x_sz, y_sz = final_masks.shape
            sz = (num_frames, num_slices, x_sz, y_sz)
            chunks = (1, num_slices, x_sz, y_sz)
            masks_zarr = zarr.open(mask_fname, mode='w-',
                                   shape=sz, chunks=chunks, dtype=np.uint16)
        masks_zarr[i, :, :, :] = final_masks
        # if i == start_volume:
        #     tiff.imwrite(mask_fname,
        #                  final_masks,
        #                  append=False,
        #                  bigtiff=True)
        # else:
        #     tiff.imwrite(mask_fname,
        #                  final_masks,
        #                  append=True,
        #                  bigtiff=True)

        # metadata dictionary
        meta_df = get_metadata_dictionary(final_masks, volume)
        metadata[i] = meta_df

    # saving metadata
    with open(metadata_fname, 'wb') as meta_save:
        pickle.dump(metadata, meta_save)

    if _config['self_path'] is not None:
        edit_config(_config['self_path'], _config)

    if verbose >= 1:
        print(f'Done with segmentation pipeline! Data saved at {mask_fname}')



def perform_post_processing_2d(mask_array, img_volume, border_width_to_remove, to_remove_border=True,
                               upper_length_threshold=12, lower_length_threshold=3, verbose=0):
    """
    Performs some post-processing steps including: Splitting long neurons, removing short neurons and
    removing too large areas

    Parameters
    ----------
    mask_array : 3D numpy array
        array of segmented masks
    img_volume : 3D numpy array
        array of original image with brightness values
    border_width_to_remove : int
        within that distance to border, artefacts/masks will be removed
    to_remove_border : boolean
        if true, a certain width
    upper_length_threshold : int
        masks longer than this will be (tried to) split
    lower_length_threshold : int
        masks shorter than this will be removed
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    final_masks : 3D numpy array
        3D array of masks after post-processing

    """
    if verbose >= 1:
        print(f"Starting preprocessing with {len(np.unique(mask_array))-1} neurons")
        print("Note: not yet stitched in z")
    masks = post.remove_large_areas(mask_array, verbose=verbose)
    if verbose >= 1:
        print(f"After large area removal: {len(np.unique(masks))-1}")
    stitched_masks, df_with_centroids = post.bipartite_stitching(masks, verbose=verbose)
    if verbose >= 1:
        print(f"After stitching: {len(np.unique(stitched_masks))-1}")
    neuron_lengths = post.get_neuron_lengths_dict(stitched_masks)

    # calculate brightnesses and their global Z-plane
    brightnesses, neuron_planes = post.calc_brightness(img_volume, stitched_masks, neuron_lengths)
    # split too long neurons
    split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
        post.split_long_neurons(stitched_masks,
                                neuron_lengths,
                                brightnesses,
                                len(neuron_lengths),
                                upper_length_threshold,
                                neuron_planes,
                                verbose-1)
    if verbose >= 1:
        print(f"After splitting: {len(np.unique(split_masks))-1}")

    final_masks, final_neuron_lengths, final_brightness, final_neuron_planes, removed_neurons_list = \
        post.remove_short_neurons(split_masks,
                                  split_lengths,
                                  lower_length_threshold,
                                  split_brightnesses,
                                  split_neuron_planes)
    if verbose >= 1:
        print(f"After short neuron removal: {len(np.unique(final_masks))-1}")

    if to_remove_border is True:
        final_masks = post.remove_border(final_masks, border_width_to_remove)

    if verbose >= 1:
        print(f"After border removal: {len(np.unique(final_masks))}")
        print("Postprocessing finished")

    return final_masks


def perform_post_processing_3d(stitched_masks, img_volume, border_width_to_remove, to_remove_border=True,
                               upper_length_threshold=12, lower_length_threshold=3, verbose=0):
    """
    Performs post-processing of segmented masks. Includes: splitting long masks, removing large areas,
    removing short masks as well as removing artefacts close to the border

    Parameters
    ----------
    stitched_masks
        array of segmented masks
    img_volume : 3D numpy array
        array of original image with brightness values
    border_width_to_remove : int
        within that distance to border, artefacts/masks will be removed
    to_remove_border : boolean
        if true, a certain width
    upper_length_threshold : int
        masks longer than this will be (tried to) split
    lower_length_threshold : int
        masks shorter than this will be removed
    verbose : int
        flag for print statements. Increasing by 1, increases depth by 1

    Returns
    -------
    labels : 3D numpy array
        3D array of processed masks
    """

    stitched_masks = post.remove_large_areas(stitched_masks)
    neuron_lengths = post.get_neuron_lengths_dict(stitched_masks)

    # calculate brightnesses and their global Z-plane
    brightnesses, neuron_planes = post.calc_brightness(img_volume, stitched_masks, neuron_lengths, verbose=verbose-1)
    # split too long neurons
    split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
        post.split_long_neurons(stitched_masks,
                                neuron_lengths,
                                brightnesses,
                                len(neuron_lengths),
                                upper_length_threshold,
                                neuron_planes,
                                verbose=verbose-1)

    # remove short neurons
    final_masks, final_neuron_lengths, final_brightness, final_neuron_planes, removed_neurons_list = \
        post.remove_short_neurons(split_masks,
                                  split_lengths,
                                  lower_length_threshold,
                                  split_brightnesses,
                                  split_neuron_planes)

    if to_remove_border is True:
        final_masks = post.remove_border(final_masks, border_width_to_remove)

    return final_masks
