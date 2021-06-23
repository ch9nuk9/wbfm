import concurrent.futures
import os
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from DLC_for_WBFM.utils.feature_detection.visualize_using_dlc import build_subset_df
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords3d
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
import pandas as pd
import numpy as np
import zarr
import raster_geometry.raster
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import best_tracklet_covering


def reindex_segmentation(this_config, DEBUG=False):
    """
    Reindexes segmentation, which originally has arbitrary numbers, to reflect tracking
    """
    trace_cfg = this_config['traces_cfg']
    seg_cfg = this_config['segment_cfg']

    with safe_cd(Path(this_config['project_path']).parent):
        # Get original segmentation
        seg_fname = seg_cfg['output']['masks']
        seg_masks = zarr.open(seg_fname)

        out_fname = os.path.join("4-traces", "reindexed_masks.zarr")
        print(f"Saving masks at {out_fname}")
        new_masks = zarr.open_like(seg_masks, path=out_fname)

        # Get tracking (dataframe) with neuron names
        matches_fname = Path(trace_cfg['all_matches'])
        all_matches = pd.read_pickle(matches_fname)
        # Format: dict with i_volume -> Nx3 array of [dlc_ind, segmentation_ind, confidence] triplets

    all_lut = all_matches_to_lookup_tables(all_matches)

    # Apply lookup tables to each volume
    # Also see link for ways to speed this up:
    # https://stackoverflow.com/questions/14448763/is-there-a-convenient-way-to-apply-a-lookup-table-to-a-large-array-in-numpy
    # for i_volume, lut in tqdm(all_lut.items()):
    #     new_masks[i_volume, ...] = lut[seg_masks[i_volume, ...]]
    #     if DEBUG:
    #         print("DEBUG mode; quitting after first volume")
    #         break

    with tqdm(total=len(all_lut)) as pbar:
        def parallel_func(i):
            lut = all_lut[i]
            new_masks[i, ...] = lut[seg_masks[i, ...]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # executor.map(parallel_func, range(len(all_lut)))
            future_results = {executor.submit(parallel_func, i): i for i in range(len(all_lut))}
            for future in concurrent.futures.as_completed(future_results):
                # _ = future_results[future]
                _ = future.result()
                pbar.update(1)


def create_spherical_segmentation(this_config, sphere_radius, DEBUG=False):
    """
    Creates a new psuedo-segmentation, which is just a sphere centered on the tracking point
    """
    track_cfg = this_config['track_cfg']
    seg_cfg = this_config['segment_cfg']

    # Required if using in multiple processes
    from zarr import blosc
    blosc.use_threads = False

    with safe_cd(Path(this_config['project_path']).parent):
        # Get original segmentation, just for shaping
        seg_fname = seg_cfg['output']['masks']
        seg_masks = zarr.open(seg_fname)

        # Initialize the masks at 0
        out_fname = os.path.join("3-tracking", "segmentation_from_tracking.zarr")
        print(f"Saving masks at {out_fname}")
        new_masks = zarr.open_like(seg_masks, path=out_fname,
                           synchronizer=zarr.ThreadSynchronizer())
        mask_sz = new_masks.shape

        # Get the 3d DLC tracks
        df_fname = track_cfg['final_3d_tracks']['df_fname']
        df = pd.read_hdf(df_fname)

    neuron_names = df.columns.levels[0]
    num_frames = mask_sz[0]
    chunk_sz = new_masks.chunks

    # Generate spheres for each neuron, for all time
    # cube_sz = [3, 7, 7]
    # for ind_neuron, neuron in tqdm(enumerate(neuron_names)):
    #     this_df = df[neuron]
    #
    #     def parallel_func(i_time):
    #         # FLIP XY
    #         z, y, x = [int(this_df['z'][i_time]), int(this_df['x'][i_time]), int(this_df['y'][i_time])]
    #         # this_shape = np.array(raster_geometry.raster.sphere(chunk_sz[1:], radius=sphere_radius, position=position))
    #         # new_masks[i_time, ...] = ind_neuron * this_shape
    #         # Instead do a cube (just for visualization)
    #         z, x, y = get_crop_coords3d([z, x, y], cube_sz, chunk_sz)
    #         new_masks[i_time, z[0]:z[-1]+1, x[0]:x[-1]+1, y[0]:y[-1]+1] = ind_neuron
    #
    #     with tqdm(total=num_frames, leave=False) as pbar:
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    #             future_results = {executor.submit(parallel_func, i): i for i in range(num_frames)}
    #             for future in concurrent.futures.as_completed(future_results):
    #                 _ = future.result()
    #                 pbar.update(1)

    cube_sz = [3, 7, 7]

    def create_cube(i_time, ind_neuron, neuron):
        # Inner loop: one time and one neuron
        this_df = df[neuron]
        # FLIP XY
        z, y, x = [int(this_df['z'][i_time]), int(this_df['x'][i_time]), int(this_df['y'][i_time])]
        # Instead do a cube (just for visualization)
        z, x, y = get_crop_coords3d([z, x, y], cube_sz, chunk_sz)
        new_masks[i_time, z[0]:z[-1]+1, x[0]:x[-1]+1, y[0]:y[-1]+1] = ind_neuron

    def process_single_volume(i_time):
        # Outer loop: process one full volume
        for i_neuron, neuron in enumerate(neuron_names):
            create_cube(i_time, i_neuron, neuron)

    # Instead do a process pool that finishes one file at a time
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        with tqdm(total=num_frames) as progress:
            futures = []

            for t in range(num_frames):
                future = pool.submit(process_single_volume, t)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            results = []
            for future in futures:
                result = future.result()
                results.append(result)

    # Reset to automatic detection
    blosc.use_threads = None


def all_matches_to_lookup_tables(all_matches):
    """
    Convert a dictionary of match arrays into a lookup table

    Match format:
        all_matches[i_volume] = [[new_ind, old_ind],...]
        Shape: Nx2+ (can be >2)

    Output usage:
        lut = all_lut[i]
        new_masks[i, ...] = lut[old_masks[i, ...]]
    """
    # Convert dataframe to lookup tables, per volume
    # Note: if not all neurons are in the dataframe, then they are set to 0
    all_lut = defaultdict(list)
    for i_volume, match in all_matches.items():
        lut = np.zeros(1000, dtype=int)  # TODO: Should be more than the maximum local index
        # TODO: are the matches always the same length?
        try:
            dlc_ind = np.array(match)[:, 0].astype(int)
            seg_ind = np.array(match)[:, 1].astype(int)
            lut[seg_ind] = dlc_ind  # Raw indices of the lut should match the local index
        except IndexError:
            # Some volumes may be empty
            pass
        all_lut[i_volume] = lut
    return all_lut


def reindex_segmentation_only_training_data(this_config, DEBUG=False):
    """
    Using tracklets and full segmentation, produces a small video (zarr) with neurons colored by track
    """

    # Get ALL matches to the segmentation, then subset
    # Also get segmentation metadata
    with safe_cd(this_config['project_dir']):
        fname = os.path.join('2-training_data', 'raw', 'clust_df_dat.pickle')
        df = pd.read_pickle(fname)

        # Get the frames chosen as training data, or recalculate
        which_frames = _get_or_recalculate_which_frames(DEBUG, df, this_config)

        # Build a sub-df with only the relevant neurons; all slices
        # Todo: connect up to actually tracked z slices?
        subset_opt = {'which_z': None,
                      'max_z_dist': None,
                      'verbose': 1}
        subset_df = build_subset_df(df, which_frames, **subset_opt)

        fname = this_config['segment_cfg']['output']['metadata']
        with open(fname, 'rb') as f:
            segmentation_metadata = pickle.load(f)

        fname = this_config['segment_cfg']['output']['masks']
        masks = zarr.open(fname)

    # Convert dataframe to matches per frame
    all_matches = {}
    for i, i_frame in tqdm(enumerate(which_frames)):
        matches = []
        for _, neuron_df in subset_df.iterrows():
            i_tracklet = neuron_df['all_ind_local'][i].astype(int)
            seg_ind = segmentation_metadata[i_frame].index[i_tracklet].astype(int)
            global_ind = neuron_df['clust_ind'] + 1
            matches.append([global_ind, seg_ind])
        all_matches[i_frame] = np.array(matches)

    # Reindex using look-up table
    all_lut = all_matches_to_lookup_tables(all_matches)

    # Initialize new array
    new_sz = list(masks.shape)
    new_sz[0] = len(which_frames)
    out_fname = os.path.join(this_config['project_dir'], '2-training_data', 'reindexed_masks.zarr')
    new_masks = zarr.open_like(masks, path=out_fname, shape=new_sz)

    # Reindex; this automatically writes to disk
    for i, (i_volume, lut) in tqdm(enumerate(all_lut.items())):
        new_masks[i, ...] = lut[masks[i_volume, ...]]

    # Note: automatically saves


def _get_or_recalculate_which_frames(DEBUG, df, this_config):
    try:
        which_frames = this_config['track_cfg']['training_data_3d']['which_frames']
    except KeyError:
        which_frames = None
    if which_frames is None:
        # Choose a subset of frames with enough tracklets
        num_frames_needed = this_config['track_cfg']['training_data_3d']['num_training_frames']
        tracklet_opt = {'num_frames_needed': num_frames_needed,
                        'num_frames': this_config['dataset_params']['num_frames'],
                        'verbose': 1}
        if DEBUG:
            tracklet_opt['num_frames_needed'] = 2
        which_frames, _ = best_tracklet_covering(df, **tracklet_opt)
    return which_frames
