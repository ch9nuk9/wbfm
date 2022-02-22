import logging
import os
import threading
from DLC_for_WBFM.utils.general.custom_errors import NoMatchesError
from DLC_for_WBFM.utils.projects.utils_filenames import add_name_suffix
from DLC_for_WBFM.utils.projects.utils_project_status import check_all_needed_data_for_step
from numcodecs import blosc

import segmentation.util.utils_postprocessing as post
import numpy as np
from tqdm import tqdm
# preprocessing
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig, ConfigFileWithProjectContext
from DLC_for_WBFM.utils.general.preprocessing.utils_tif import perform_preprocessing
from DLC_for_WBFM.utils.projects.utils_project import edit_config
# metadata
from segmentation.util.utils_config_files import _unpack_config_file
from segmentation.util.utils_metadata import get_metadata_dictionary, calc_metadata_full_video
import zarr
import concurrent.futures


def segment_video_using_config_3d(segment_cfg: ConfigFileWithProjectContext,
                                  project_cfg: ModularProjectConfig,
                                  continue_from_frame: int =None,
                                  DEBUG: bool = False) -> None:
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

    frame_list, mask_fname, metadata_fname, num_frames, stardist_model_name, verbose, video_path, _, all_bounding_boxes = _unpack_config_file(
        segment_cfg, project_cfg, DEBUG)

    # Open the file
    if not video_path.endswith('.zarr'):
        raise ValueError("Non-zarr usage has been deprecated")
    video_dat = zarr.open(video_path)

    sd_model = initialize_stardist_model(stardist_model_name, verbose)
    # Do first volume outside the parallelization loop to initialize keras and zarr
    masks_zarr = _do_first_volume3d(frame_list, mask_fname, num_frames,
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

    _segment_full_video_3d(segment_cfg, frame_list, mask_fname, num_frames, verbose, video_dat,
                           segmentation_options, continue_from_frame)

    calc_metadata_full_video(frame_list, masks_zarr, video_dat, metadata_fname)


def _segment_full_video_3d(_config: dict, frame_list: list, mask_fname: str, num_frames: int, verbose: int,
                           video_dat: zarr.Array,
                           opt: dict, continue_from_frame: int) -> None:
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

    if _config.config.get('self_path', None) is not None:
        edit_config(_config.config['self_path'], _config.config)
    if verbose >= 1:
        print(f'Done with segmentation pipeline! Mask data saved at {mask_fname}')

##
## 2d pipeline (stitch to get 3d)
##


def segment_video_using_config_2d(segment_cfg: ConfigFileWithProjectContext,
                                  project_cfg: ModularProjectConfig,
                                  continue_from_frame: int = None,
                                  DEBUG: bool = False) -> None:
    """
    Full pipeline based on only a config file

    See segment2d.py for parameter documentation
    """

    frame_list, mask_fname, metadata_fname, num_frames, stardist_model_name, verbose, video_path, zero_out_borders, all_bounding_boxes = _unpack_config_file(
        segment_cfg, project_cfg, DEBUG)

    # Open the file
    if not video_path.endswith('.zarr'):
        raise ValueError("Non-zarr usage has been deprecated")
    video_dat = zarr.open(video_path)

    sd_model = initialize_stardist_model(stardist_model_name, verbose)
    # Do first volume outside the parallelization loop to initialize keras and zarr
    opt_postprocessing = segment_cfg.config['postprocessing_params']  # Unique to 2d
    if verbose > 1:
        print("Postprocessing settings: ")
        print(opt_postprocessing)
    # Force BLOSC (compression within zarr) to not be multi-threaded
    # On large jobs (not remote, not dask) I get this error: https://github.com/pangeo-data/pangeo/issues/196
    os.environ["BLOSC_NTHREADS"] = "1"
    # See also: https://zarr.readthedocs.io/en/stable/tutorial.html#tutorial-tips-blosc
    blosc.use_threads = False
    masks_zarr = _do_first_volume2d(frame_list, mask_fname, num_frames,
                                    sd_model, verbose, video_dat, zero_out_borders,
                                    all_bounding_boxes,
                                    continue_from_frame, opt_postprocessing)

    # Main function
    segmentation_options = {'masks_zarr': masks_zarr, 'opt_postprocessing': opt_postprocessing,
                            'sd_model': sd_model, 'verbose': verbose, 'zero_out_borders': zero_out_borders,
                            'all_bounding_boxes': all_bounding_boxes}

    # Will always be at least continuing after the first frame
    if continue_from_frame is None:
        continue_from_frame = 1
    else:
        continue_from_frame += 1
        print(f"Continuing from frame {continue_from_frame}")

    _segment_full_video_2d(segment_cfg, frame_list, mask_fname, num_frames, verbose, video_dat,
                           segmentation_options, continue_from_frame)

    # Same 2d and 3d
    calc_metadata_full_video(frame_list, masks_zarr, video_dat, metadata_fname)


def initialize_stardist_model(stardist_model_name, verbose):
    from segmentation.util.utils_model import get_stardist_model
    sd_model = get_stardist_model(stardist_model_name, verbose=verbose - 1)
    # Not fully working for multithreaded scenario
    # Discussion about finalizing: https://stackoverflow.com/questions/40850089/is-keras-thread-safe/43393252#43393252
    # Dicussion about making the predict function: https://github.com/jaromiru/AI-blog/issues/2
    sd_model.keras_model.make_predict_function()
    return sd_model


def _segment_full_video_2d(segment_cfg: ConfigFileWithProjectContext,
                           frame_list: list, mask_fname: str, num_frames: int, verbose: int,
                           video_dat: zarr.Array,
                           opt: dict, continue_from_frame: int) -> None:

    # Parallel version: threading
    keras_lock = threading.Lock()
    read_lock = threading.Lock()
    opt['keras_lock'] = keras_lock
    opt['read_lock'] = read_lock

    with tqdm(total=num_frames - continue_from_frame) as pbar:
        def parallel_func(i_both):
            i_out, i_vol = i_both
            segment_and_save2d(i_out + continue_from_frame, i_vol, video_dat=video_dat, **opt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(parallel_func, i): i for i in enumerate(frame_list[continue_from_frame:])}
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    segment_cfg.update_on_disk()
    if verbose >= 1:
        print(f'Done with segmentation pipeline! Mask data saved at {mask_fname}')


def _do_first_volume2d(frame_list: list, mask_fname: str, num_frames: int,
                       sd_model, verbose: int, video_dat: zarr.Array,
                       zero_out_borders: bool,
                       all_bounding_boxes: list = None,
                       continue_from_frame: int = None, opt_postprocessing: dict = None) -> zarr.Array:
    # Do first loop to initialize the zarr data
    if continue_from_frame is None:
        i = 0
        mode = 'w-'
    else:
        i = continue_from_frame
        # Old file MUST exist in this case
        mode = 'r+'
    i_volume = frame_list[i]
    volume = get_volume_using_bbox(all_bounding_boxes, i_volume, video_dat)

    from segmentation.util.utils_model import segment_with_stardist_2d
    final_masks = segment_with_stardist_2d(volume, sd_model, zero_out_borders, verbose=verbose - 1)
    _, num_slices, x_sz, y_sz = video_dat.shape
    masks_zarr = _create_or_continue_zarr(mask_fname, num_frames, num_slices, x_sz, y_sz, mode=mode)
    final_masks = perform_post_processing_2d(final_masks,
                                             volume,
                                             **opt_postprocessing,
                                             verbose=verbose - 1)
    save_volume_using_bbox(all_bounding_boxes, final_masks, i, i_volume, masks_zarr)
    return masks_zarr


def get_volume_using_bbox(all_bounding_boxes, i_volume, video_dat):
    if all_bounding_boxes is None:
        volume = video_dat[i_volume, ...]
    else:
        bbox = all_bounding_boxes[i_volume]
        volume = video_dat[i_volume, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return volume


def _do_first_volume3d(frame_list: list, mask_fname: str, num_frames: int,
                       sd_model, verbose: int, video_dat: zarr.Array,
                       continue_from_frame: int = None) -> zarr.Array:
    # Do first loop to initialize the zarr data
    if continue_from_frame is None:
        i = 0
        mode = 'w-'
    else:
        i = continue_from_frame
        # Old file MUST exist in this case
        mode = 'r+'
    i_volume = frame_list[i]
    volume = video_dat[i_volume, ...]
    from segmentation.util.utils_model import segment_with_stardist_3d
    final_masks = segment_with_stardist_3d(volume, sd_model, verbose=verbose - 1)
    _, num_slices, x_sz, y_sz = video_dat.shape
    masks_zarr = _create_or_continue_zarr(mask_fname, num_frames, num_slices, x_sz, y_sz, mode=mode)
    # Add masks to zarr file; automatically saves
    masks_zarr[i, :, :, :] = final_masks
    return masks_zarr


def _create_or_continue_zarr(mask_fname, num_frames, num_slices, x_sz, y_sz, mode='w-'):
    """Creates a new zarr file of the correct file, or, if it already exists, check for a stopping point"""
    sz = (num_frames, num_slices, x_sz, y_sz)
    chunks = (1, num_slices, x_sz, y_sz)
    print(f"Opening zarr at: {mask_fname}")
    try:
        masks_zarr = zarr.open(mask_fname, mode=mode,
                               shape=sz, chunks=chunks, dtype=np.uint16,
                               fill_value=0,
                               synchronizer=zarr.ThreadSynchronizer())
    except zarr.errors.ContainsArrayError as e:
        print("???????????????????????????????????????????????????????")
        print("Array exists; did you mean to pass continue_from_frame?")
        print("???????????????????????????????????????????????????????")
        raise e
    return masks_zarr


def segment_and_save3d(i, i_volume, masks_zarr,
                       sd_model, verbose, video_dat, keras_lock=None, read_lock=None):
    volume = video_dat[i_volume, ...]
    # volume = _get_and_prepare_volume(i_volume, num_slices, preprocessing_settings, video_path, read_lock=read_lock)
    from segmentation.util.utils_model import segment_with_stardist_3d
    if keras_lock is None:
        final_masks = segment_with_stardist_3d(volume, sd_model, verbose=verbose - 1)
    else:
        with keras_lock:  # Keras is not thread-safe in the end
            final_masks = segment_with_stardist_3d(volume, sd_model, verbose=verbose - 1)

    # save_masks_and_metadata(final_masks, i, i_volume, masks_zarr, metadata, volume)
    masks_zarr[i, :, :, :] = final_masks


def segment_and_save2d(i, i_volume, masks_zarr, opt_postprocessing,
                       zero_out_borders,
                       all_bounding_boxes,
                       sd_model, verbose, video_dat, keras_lock=None, read_lock=None):
    volume = get_volume_using_bbox(all_bounding_boxes, i_volume, video_dat)
    from segmentation.util.utils_model import segment_with_stardist_2d
    if keras_lock is None:
        segmented_masks = segment_with_stardist_2d(volume, sd_model, zero_out_borders, verbose=verbose - 1)
    else:
        with keras_lock:  # Keras is not thread-safe in the end
            segmented_masks = segment_with_stardist_2d(volume, sd_model, zero_out_borders, verbose=verbose - 1)

    final_masks = perform_post_processing_2d(segmented_masks,
                                             volume,
                                             **opt_postprocessing,
                                             verbose=verbose - 1)
    save_volume_using_bbox(all_bounding_boxes, final_masks, i, i_volume, masks_zarr)


def save_volume_using_bbox(all_bounding_boxes, final_masks, i, i_volume, masks_zarr):
    if all_bounding_boxes is None:
        masks_zarr[i, :, :, :] = final_masks
    else:
        bbox = all_bounding_boxes[i_volume]
        masks_zarr[i, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] = final_masks


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


def perform_post_processing_2d(mask_array: np.ndarray, img_volume, border_width_to_remove, to_remove_border=True,
                               upper_length_threshold=12, lower_length_threshold=3,
                               to_remove_dim_slices=False,
                               stitch_via_watershed=False,
                               min_separation=0,
                               already_stitched=False,
                               also_split_using_centroids=False,
                               verbose=0,
                               DEBUG=False):
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
    stitch_via_watershed : bool
        Default is False, which means stitching via bipartite matching and a lot of post-processing
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

    if not stitch_via_watershed:
        if not already_stitched:
            try:
                stitched_masks, intermediates = post.bipartite_stitching(masks, verbose=verbose)
            except NoMatchesError:
                logging.warning("No between-plane matches found; skipping this volume")
                return np.zeros_like(masks)
        else:
            stitched_masks = masks.copy()
        if verbose >= 1:
            print(f"After stitching: {len(np.unique(stitched_masks)) - 1}")
        neuron_lengths = post.get_neuron_lengths_dict(stitched_masks)

        # calculate brightnesses and their global Z-plane
        brightnesses, neuron_planes, neuron_centroids = post.calc_brightness(img_volume, stitched_masks)
        # split too long neurons
        current_global_neuron = len(neuron_lengths)
        split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
            post.split_long_neurons(stitched_masks,
                                    neuron_lengths,
                                    brightnesses,
                                    neuron_centroids,
                                    current_global_neuron,
                                    upper_length_threshold,
                                    neuron_planes,
                                    min_separation,
                                    also_split_using_centroids,
                                    verbose=verbose - 1)
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
    else:
        if verbose >= 1:
            print("Stitching using watershed")
        final_masks = post.stitch_via_watershed(masks, img_volume)

    if to_remove_border:
        final_masks = post.remove_border(final_masks, border_width_to_remove)

    if verbose >= 1:
        print(f"After border removal: {len(np.unique(final_masks))}")
        print("Postprocessing finished")

    if DEBUG:
        from DLC_for_WBFM.utils.projects.utils_debugging import shelve_full_workspace
        fname = 'stardist_2d_postprocessing.out'
        shelve_full_workspace(fname, list(dir()), locals())

    return final_masks


def resplit_masks_in_z_from_config(segment_cfg: ConfigFileWithProjectContext,
                                   project_cfg: ModularProjectConfig,
                                   continue_from_frame: int = None, DEBUG=False) -> None:
    """
    Similar to segment_full_video_2d, but this assumes a previously segmented video

    """

    frame_list, mask_fname, metadata_fname, num_frames, _, verbose, video_path, zero_out_borders, all_bounding_boxes = _unpack_config_file(
        segment_cfg, project_cfg, DEBUG)

    # Get data: needs both segmentation and raw video
    check_all_needed_data_for_step(project_cfg.self_path, 2)
    masks_old = zarr.open(mask_fname, mode='r')
    # masks_old = np.array(masks_zarr[:num_frames, ...])  # TEST

    # Do not overwrite old file
    new_fname = str(add_name_suffix(mask_fname, '1'))
    masks_zarr = zarr.open_like(masks_old, path=new_fname)

    video_dat = zarr.open(video_path, mode='r')

    opt_postprocessing = segment_cfg.config['postprocessing_params']  # Unique to 2d
    opt = {'opt_postprocessing': opt_postprocessing,
           'verbose': verbose,
           'all_bounding_boxes': all_bounding_boxes}
    read_lock = threading.Lock()
    write_lock = threading.Lock()
    opt['read_lock'] = read_lock
    opt['write_lock'] = write_lock
    if continue_from_frame is None:
        # Note that this does NOT have a separate 'do first volume' function
        continue_from_frame = 0

    for i_out, i_vol in enumerate(tqdm(frame_list[continue_from_frame:])):
        _only_postprocess2d(i_out + continue_from_frame, i_vol, video_dat=video_dat,
                            masks_old=masks_old, masks_zarr=masks_zarr, **opt)

    #... GENUINELY NO IDEA WHY THREADS DON'T WORK HERE
    # Actually split
    # with tqdm(total=num_frames - continue_from_frame) as pbar:
    #     def parallel_func(i_both):
    #         i_out, i_vol = i_both
    #         _only_postprocess2d(i_out + continue_from_frame, i_vol, video_dat=video_dat, masks_zarr=masks_zarr,
    #                             masks_old=masks_old, **opt)
    #
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    #         futures = {executor.submit(parallel_func, i): i for i in enumerate(frame_list[continue_from_frame:])}
    #         for future in concurrent.futures.as_completed(futures):
    #             future.result()
    #             pbar.update(1)

    # Update metadata
    segment_cfg.update_on_disk()
    if verbose >= 1:
        print(f'Done with segmentation pipeline! Mask data saved at {mask_fname}')

    # Same 2d and 3d
    calc_metadata_full_video(frame_list, masks_zarr, video_dat, metadata_fname)


def _only_postprocess2d(i, i_volume, masks_zarr, opt_postprocessing,
                       all_bounding_boxes, read_lock, write_lock, masks_old,
                       verbose, video_dat):
    volume = get_volume_using_bbox(all_bounding_boxes, i_volume, video_dat)
    # Read mask directly from previously segmented volume, but copy it
    with read_lock:
        segmented_masks = np.array(masks_old[i_volume, :, :, :])
        # segmented_masks = np.array(masks_zarr[i, :, :, :])
    if verbose >= 1:
        print(f"Analyzing {i}, {i_volume}")
        if verbose >= 3:
            print(f"Mean values: {np.mean(segmented_masks)}, {np.mean(volume)}")
    final_masks = perform_post_processing_2d(segmented_masks,
                                             volume,
                                             **opt_postprocessing,
                                             verbose=verbose - 1)
    with read_lock:
        save_volume_using_bbox(all_bounding_boxes, final_masks, i_volume, i_volume, masks_zarr)


def perform_post_processing_3d(stitched_masks, img_volume, border_width_to_remove, to_remove_border=True,
                               upper_length_threshold=12, lower_length_threshold=3, split_using_centroids_not_brightness=False, verbose=0):
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
    brightnesses, neuron_planes, neuron_centroids = post.calc_brightness(img_volume, stitched_masks, verbose=verbose - 1)
    # split too long neurons
    split_masks, split_lengths, split_brightnesses, current_global_neuron, split_neuron_planes = \
        post.split_long_neurons(stitched_masks,
                                neuron_lengths,
                                brightnesses,
                                neuron_centroids,
                                len(neuron_lengths),
                                upper_length_threshold,
                                neuron_planes,
                                split_using_centroids_not_brightness,
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
