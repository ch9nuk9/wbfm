import threading
from multiprocessing import Manager

import tifffile
import segmentation.util.utils_postprocessing as post
import numpy as np
from tqdm import tqdm
import pickle
# preprocessing
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
from DLC_for_WBFM.utils.preprocessing.utils_tif import perform_preprocessing
from DLC_for_WBFM.utils.projects.utils_project import edit_config
# metadata
from segmentation.util.utils_metadata import get_metadata_dictionary
from segmentation.util.utils_paths import get_output_fnames
import zarr
from segmentation.util.utils_model import segment_with_stardist_2d, segment_with_stardist_3d
from segmentation.util.utils_model import get_stardist_model
import concurrent.futures


def segment_video_using_config_3d(_config, continue_from_frame=None):
    """

    Parameters
    ----------
    _config - dict
        Parameters as loaded from a .yaml file. See segment3d.py for documentation
    continue_from_frame - int or None
        For example, if the segmentation crashed, then continue from this frame instead of starting anew

    Returns
    -------
    Saves masks and metadata in the project subfolder 1-segmentation

    """

    frame_list, mask_fname, metadata_fname, num_frames, num_slices, stardist_model_name, verbose, video_path = _unpack_config_file(
        _config)

    # Open the file
    if not video_path.endswith('.zarr'):
        raise ValueError("Non-zarr usage has been deprecated")
    video_dat = zarr.open(video_path)

    sd_model = get_stardist_model(stardist_model_name, verbose=verbose - 1)
    # Not fully working for multithreaded scenario
    # Discussion about finalizing: https://stackoverflow.com/questions/40850089/is-keras-thread-safe/43393252#43393252
    # Dicussion about making the predict function: https://github.com/jaromiru/AI-blog/issues/2
    sd_model.keras_model.make_predict_function()
    # Do first volume outside the parallelization loop to initialize keras and zarr
    masks_zarr = _do_first_volume3d(frame_list, mask_fname, num_frames, num_slices,
                                    sd_model, verbose, video_dat, continue_from_frame)
    # Main function
    segmentation_options = {'masks_zarr': masks_zarr,
                            'sd_model': sd_model, 'verbose': verbose}

    # Will always be at least continuing after the first frame
    if continue_from_frame is None:
        continue_from_frame = 1
    else:
        continue_from_frame += 1
        print(f"Continuing from frame {continue_from_frame}")

    _segment_full_video_3d(_config, frame_list, mask_fname, num_frames, verbose, video_dat,
                           segmentation_options, continue_from_frame)

    calc_metadata_full_video_3d(frame_list, masks_zarr, video_dat, metadata_fname)


def calc_metadata_full_video_3d(frame_list, masks_zarr, video_dat, metadata_fname):
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


def _segment_full_video_3d(_config, frame_list, mask_fname, num_frames, verbose, video_dat,
                           opt, continue_from_frame):
    # with tifffile.TiffFile(video_path) as video_stream:
    #     for i_rel, i_abs in tqdm(enumerate(frame_list[1:]), total=len(frame_list) - 1):
    #         segment_and_save3d(i_rel + 1, i_abs, **opt, video_path=video_stream)
    # Parallel version: threading
    keras_lock = threading.Lock()
    read_lock = threading.Lock()
    opt['keras_lock'] = keras_lock
    opt['read_lock'] = read_lock

    with tqdm(total=num_frames - continue_from_frame) as pbar:
        def parallel_func(i_both):
            i_out, i_vol = i_both
            segment_and_save3d(i_out + continue_from_frame, i_vol, video_dat=video_dat, **opt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(parallel_func, i): i for i in enumerate(frame_list[continue_from_frame:])}
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    if _config.get('self_path', None) is not None:
        edit_config(_config['self_path'], _config)
    if verbose >= 1:
        print(f'Done with segmentation pipeline! Mask data saved at {mask_fname}')


def segment_video_using_config_2d(_config):
    """
    Full pipeline based on only a config file

    See segment2d.py for parameter documentation
    """

    frame_list, mask_fname, metadata, metadata_fname, num_frames, num_slices, preprocessing_settings, stardist_model_name, verbose, video_path = _unpack_config_file(
        _config)

    # Unique to 2d
    opt_postprocessing = _config['postprocessing_params']

    # get stardist model object
    _segment_full_video_2d(_config, frame_list, mask_fname, metadata, metadata_fname, num_frames, num_slices,
                           opt_postprocessing, preprocessing_settings, stardist_model_name, verbose, video_path)

    # _calc_metadata_full_video_2d(frame_list, masks_zarr, metadata, num_slices, preprocessing_settings, video_path)


# def _calc_metadata_full_video_2d(frame_list, masks_zarr, metadata, num_slices, preprocessing_settings, video_path):
#     # Loop again in order to calculate metadata and possibly postprocess
#     with tifffile.TiffFile(video_path) as video_stream:
#         for i_rel, i_abs in tqdm(enumerate(frame_list), total=len(frame_list)):
#             masks = masks_zarr[i_rel, :, :, :]
#             # TODO: Use a disk-saved preprocessing artifact instead of recalculating
#             volume = _get_and_prepare_volume(i_abs, num_slices, preprocessing_settings, video_path=video_stream)
#
#             metadata[i_abs] = get_metadata_dictionary(masks, volume)


def _unpack_config_file(_config):
    # Initializing variables
    start_volume = _config['dataset_params']['start_volume']
    num_frames = _config['dataset_params']['num_frames']
    if _config['DEBUG']:
        num_frames = 1
    num_slices = _config['dataset_params']['num_slices']
    frame_list = list(range(start_volume, start_volume + num_frames))
    video_path = _config['video_path']
    mask_fname, metadata_fname = get_output_fnames(video_path, _config)
    # Save settings
    _config['output']['masks'] = mask_fname
    _config['output']['metadata'] = metadata_fname
    verbose = _config['verbose']
    stardist_model_name = _config['segmentation_params']['stardist_model_name']
    return frame_list, mask_fname, metadata_fname, num_frames, num_slices, stardist_model_name, verbose, video_path


def _segment_full_video_2d(_config, frame_list, mask_fname, metadata, metadata_fname, num_frames, num_slices,
                           opt_postprocessing, preprocessing_settings, stardist_model_name, verbose, video_path):
    sd_model = get_stardist_model(stardist_model_name, verbose=verbose - 1)
    # Not fully working for multithreaded scenario
    # Discussion about finalizing: https://stackoverflow.com/questions/40850089/is-keras-thread-safe/43393252#43393252
    # Dicussion about making the predict function: https://github.com/jaromiru/AI-blog/issues/2
    sd_model.keras_model.make_predict_function()
    # Do first volume outside the parallelization loop to initialize keras and zarr
    # Possibly unnecessary
    masks_zarr = _do_first_volume2d(frame_list, mask_fname, metadata, num_frames, num_slices, opt_postprocessing,
                                    preprocessing_settings, sd_model, verbose, video_path)
    # Main function
    opt = {'masks_zarr': masks_zarr, 'metadata': metadata, 'num_slices': num_slices,
           'opt_postprocessing': opt_postprocessing, 'preprocessing_settings': preprocessing_settings,
           'sd_model': sd_model, 'verbose': verbose}
    # Sequential version
    # with tifffile.TiffFile(video_path) as video_stream:
    #     for i_rel, i_abs in tqdm(enumerate(frame_list[1:]), total=len(frame_list)-1):
    #         segment_and_save(i_rel + 1, i_abs, **opt, video_path=video_stream)
    # Parallel version: threading
    keras_lock = threading.Lock()
    read_lock = threading.Lock()
    opt['keras_lock'] = keras_lock
    opt['read_lock'] = read_lock
    with tqdm(total=num_frames - 1) as pbar:
        with tifffile.TiffFile(video_path) as video_stream:
            def parallel_func(i_both):
                i_out, i_vol = i_both
                segment_and_save2d(i_out + 1, i_vol, video_path=video_stream, **opt)

            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                futures = {executor.submit(parallel_func, i): i for i in enumerate(frame_list[1:])}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)
    # saving metadata and settings
    with open(metadata_fname, 'wb') as meta_save:
        pickle.dump(metadata, meta_save)
    if _config.get('self_path', None) is not None:
        edit_config(_config['self_path'], _config)
    if verbose >= 1:
        print(f'Done with segmentation pipeline! Mask data saved at {mask_fname}')


def _do_first_volume2d(frame_list, mask_fname, metadata, num_frames, num_slices, opt_postprocessing,
                       preprocessing_settings, sd_model, verbose, video_path):
    # Do first loop to initialize the zarr data
    i = 0
    i_volume = frame_list[i]
    volume = _get_and_prepare_volume(i_volume, num_slices, preprocessing_settings, video_path)
    final_masks = segment_with_stardist_2d(volume, sd_model, verbose=verbose - 1)
    _, x_sz, y_sz = final_masks.shape
    masks_zarr = _create_or_continue_zarr(mask_fname, num_frames, num_slices, x_sz, y_sz)
    final_masks = perform_post_processing_2d(final_masks,
                                             volume,
                                             **opt_postprocessing,
                                             verbose=verbose - 1)
    # Add masks to zarr file; automatically saves
    # masks_zarr[i, :, :, :] = final_masks
    save_masks_and_metadata(final_masks, i, i_volume, masks_zarr, metadata, volume)
    return masks_zarr


def _do_first_volume3d(frame_list, mask_fname, num_frames, num_slices,
                       sd_model, verbose, video_dat, continue_from_frame=None):
    # Do first loop to initialize the zarr data
    if continue_from_frame is None:
        i = 0
        mode = 'w-'
    else:
        i = continue_from_frame
        # Old file MUST exist in this case
        mode = 'r+'
    i_volume = frame_list[i]
    volume = video_dat[i, ...]
    final_masks = segment_with_stardist_3d(volume, sd_model, verbose=verbose - 1)
    _, x_sz, y_sz = final_masks.shape
    masks_zarr = _create_or_continue_zarr(mask_fname, num_frames, num_slices, x_sz, y_sz, mode=mode)
    # Add masks to zarr file; automatically saves
    masks_zarr[i, :, :, :] = final_masks
    return masks_zarr


def _create_or_continue_zarr(mask_fname, num_frames, num_slices, x_sz, y_sz, mode='w-'):
    """Creates a new zarr file of the correct file, or, if it already exists, check for a stopping point"""
    sz = (num_frames, num_slices, x_sz, y_sz)
    chunks = (1, num_slices, x_sz, y_sz)
    masks_zarr = zarr.open(mask_fname, mode=mode,
                           shape=sz, chunks=chunks, dtype=np.uint16,
                           fill_value=0,
                           synchronizer=zarr.ThreadSynchronizer())
    return masks_zarr


def segment_and_save2d(i, i_volume, masks_zarr, metadata, num_slices, opt_postprocessing, preprocessing_settings,
                       sd_model, verbose, video_path, keras_lock, read_lock):
    volume = _get_and_prepare_volume(i_volume, num_slices, preprocessing_settings, video_path, read_lock=read_lock)
    with keras_lock:  # Keras is not thread-safe in the end
        segmented_masks = segment_with_stardist_2d(volume, sd_model, verbose=verbose - 1)
    # process masks: remove large areas, stitch, split long neurons, remove short neurons
    final_masks = perform_post_processing_2d(segmented_masks,
                                             volume,
                                             **opt_postprocessing,
                                             verbose=verbose - 1)
    save_masks_and_metadata(final_masks, i, i_volume, masks_zarr, metadata, volume)
    # masks_zarr[i, :, :, :] = final_masks


def segment_and_save3d(i, i_volume, masks_zarr,
                       sd_model, verbose, video_dat, keras_lock=None, read_lock=None):
    volume = video_dat[i_volume, ...]
    # volume = _get_and_prepare_volume(i_volume, num_slices, preprocessing_settings, video_path, read_lock=read_lock)
    if keras_lock is None:
        final_masks = segment_with_stardist_3d(volume, sd_model, verbose=verbose - 1)
    else:
        with keras_lock:  # Keras is not thread-safe in the end
            final_masks = segment_with_stardist_3d(volume, sd_model, verbose=verbose - 1)

    # save_masks_and_metadata(final_masks, i, i_volume, masks_zarr, metadata, volume)
    masks_zarr[i, :, :, :] = final_masks


def _get_and_prepare_volume(i, num_slices, preprocessing_settings, video_path, read_lock=None):
    # use get single volume function from charlie
    import_opt = {'which_vol': i, 'num_slices': num_slices, 'alpha': 1.0, 'dtype': 'uint16'}
    if read_lock is None:
        volume = get_single_volume(video_path, **import_opt)
    else:
        with read_lock:
            volume = get_single_volume(video_path, **import_opt)
    volume = perform_preprocessing(volume, preprocessing_settings)
    return volume


def save_masks_and_metadata(final_masks, i, i_volume, masks_zarr, metadata, volume):
    # Add masks to zarr file; automatically saves
    masks_zarr[i, :, :, :] = final_masks
    # metadata dictionary; also modified by reference
    meta_df = get_metadata_dictionary(final_masks, volume)
    metadata[i_volume] = meta_df


def perform_post_processing_2d(mask_array, img_volume, border_width_to_remove, to_remove_border=True,
                               upper_length_threshold=12, lower_length_threshold=3,
                               to_remove_dim_slices=False,
                               verbose=0):
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
    to_remove_dim_slices : bool
        Before stitching, removes stardist segments that are too dim
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1

    Returns
    -------
    final_masks : 3D numpy array
        3D array of masks after post-processing

    """
    if verbose >= 1:
        print(f"Starting preprocessing with {len(np.unique(mask_array)) - 1} neurons")
        print("Note: not yet stitched in z")
    masks = post.remove_large_areas(mask_array, verbose=verbose)
    if to_remove_dim_slices:
        masks = post.remove_dim_slices(masks, img_volume, verbose=verbose)
    if verbose >= 1:
        print(f"After large area removal: {len(np.unique(masks)) - 1}")
    stitched_masks, intermediates = post.bipartite_stitching(masks, verbose=verbose)
    if verbose >= 1:
        print(f"After stitching: {len(np.unique(stitched_masks)) - 1}")
    neuron_lengths = post.get_neuron_lengths_dict(stitched_masks)

    # calculate brightnesses and their global Z-plane
    brightnesses, neuron_planes = post.calc_brightness(img_volume, stitched_masks, neuron_lengths)
    # split too long neurons
    current_global_neuron = len(neuron_lengths)
    split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
        post.split_long_neurons(stitched_masks,
                                neuron_lengths,
                                brightnesses,
                                current_global_neuron,
                                upper_length_threshold,
                                neuron_planes,
                                verbose - 1)
    if verbose >= 1:
        print(f"After splitting: {len(np.unique(split_masks)) - 1}")

    final_masks, final_neuron_lengths, final_brightness, final_neuron_planes, removed_neurons_list = \
        post.remove_short_neurons(split_masks,
                                  split_lengths,
                                  lower_length_threshold,
                                  split_brightnesses,
                                  split_neuron_planes)
    if verbose >= 1:
        print(f"After short neuron removal: {len(np.unique(final_masks)) - 1}")

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
    brightnesses, neuron_planes = post.calc_brightness(img_volume, stitched_masks, neuron_lengths, verbose=verbose - 1)
    # split too long neurons
    split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
        post.split_long_neurons(stitched_masks,
                                neuron_lengths,
                                brightnesses,
                                len(neuron_lengths),
                                upper_length_threshold,
                                neuron_planes,
                                verbose=verbose - 1)

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


##
## Also just for metadata calculation
##

def recalculate_metadata_from_config(_config, DEBUG=False):
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

    frame_list, mask_fname, metadata_fname, num_frames, num_slices, stardist_model_name, verbose, video_path = _unpack_config_file(
        _config)

    masks_zarr = zarr.open(_config['output']['masks'])
    video_dat = zarr.open(video_path)

    if DEBUG:
        frame_list = frame_list[:2]

    calc_metadata_full_video_3d(frame_list, masks_zarr, video_dat, metadata_fname)


