import concurrent.futures
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.projects.utils_filepaths import config_file_with_project_context, modular_project_config
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import build_subset_df_from_tracklets, \
    get_or_recalculate_which_frames


def reindex_segmentation_using_config(traces_cfg: config_file_with_project_context,
                                      segment_cfg: config_file_with_project_context,
                                      project_cfg: modular_project_config,
                                      DEBUG=False):
    """
    Reindexes segmentation, which originally has arbitrary numbers, to reflect tracking
    """
    all_matches, raw_seg_masks, new_masks = _unpack_config_reindexing(traces_cfg, segment_cfg, project_cfg)

    reindex_segmentation(DEBUG, all_matches, raw_seg_masks, new_masks)


def reindex_segmentation(DEBUG, all_matches, seg_masks, new_masks):
    all_lut = all_matches_to_lookup_tables(all_matches)
    all_lut_keys = all_lut.keys()
    if DEBUG:
        all_lut_keys = [0, 1]
        print("DEBUG mode: only doing first 2 volumes")
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
            future_results = {executor.submit(parallel_func, i): i for i in all_lut_keys}
            for future in concurrent.futures.as_completed(future_results):
                # _ = future_results[future]
                _ = future.result()
                pbar.update(1)


def _unpack_config_reindexing(traces_cfg, segment_cfg, project_cfg):
    # Get original segmentation
    seg_fname = segment_cfg.resolve_relative_path('output_masks')
    raw_seg_masks = zarr.open(seg_fname)

    relative_path = traces_cfg.config['reindexed_masks']
    out_fname = project_cfg.resolve_path_relative_to_project(relative_path)
    print(f"Saving masks at {out_fname}")
    new_masks = zarr.open_like(raw_seg_masks, path=out_fname)

    # Get tracking (dataframe) with neuron names
    matches_fname = traces_cfg.resolve_relative_path('all_matches')
    all_matches = pd.read_pickle(matches_fname)
    # Format: dict with i_volume -> Nx3 array of [dlc_ind, segmentation_ind, confidence] triplets

    return all_matches, raw_seg_masks, new_masks


def create_spherical_segmentation(this_config, sphere_radius, DEBUG=False):
    """
    Creates a new psuedo-segmentation, which is just a sphere centered on the tracking point
    """
    track_cfg = this_config['track_cfg']
    seg_cfg = this_config['segment_cfg']

    # Required if using in multiple processes
    # from zarr import blosc
    # blosc.use_threads = False

    with safe_cd(Path(this_config['project_path']).parent):
        # Get original segmentation, just for shaping
        seg_fname = seg_cfg['output_masks']
        seg_masks = zarr.open(seg_fname)

        # Initialize the masks at 0
        out_fname = os.path.join("3-tracking", "segmentation_from_tracking.zarr")
        print(f"Saving masks at {out_fname}")
        new_masks = zarr.open_like(seg_masks, path=out_fname,
                                   synchronizer=zarr.ThreadSynchronizer())
        mask_sz = new_masks.shape

        # Get the 3d DLC tracks
        df_fname = track_cfg['final_3d_tracks_df']
        df = pd.read_hdf(df_fname)

    neuron_names = df.columns.levels[0]
    num_frames = mask_sz[0]
    chunk_sz = new_masks.chunks

    # Generate spheres for each neuron, for all time
    cube_sz = [2, 4, 4]

    def get_clipped_sizes(this_sz, sz, total_sz):
        lower_dim = int(np.clip(this_sz - sz, a_min=0, a_max=total_sz))
        upper_dim = int(np.clip(this_sz + sz + 1, a_max=total_sz, a_min=0))
        return lower_dim, upper_dim

    def parallel_func(i_time: int, ind_neuron: int, this_df: pd.DataFrame):
        # X=col, Y=row
        z, col, row = [int(this_df['z'][i_time]), int(this_df['x'][i_time]), int(this_df['y'][i_time])]
        # Instead do a cube (just for visualization)
        z0, z1 = get_clipped_sizes(z, cube_sz[0], chunk_sz[1])
        row0, row1 = get_clipped_sizes(row, cube_sz[1], chunk_sz[2])
        col0, col1 = get_clipped_sizes(col, cube_sz[2], chunk_sz[3])
        new_masks[i_time, z0:z1, row0:row1, col0:col1] = ind_neuron + 1  # Skip 0

    for ind_neuron, neuron in tqdm(enumerate(neuron_names), total=len(neuron_names)):
        # for i in tqdm(range(num_frames), total=num_frames, leave=False):
        #     parallel_func(i, this_df=df[neuron])

        with tqdm(total=num_frames, leave=False) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                future_results = {executor.submit(parallel_func, i, ind_neuron=ind_neuron, this_df=df[neuron]): i for i
                                  in range(num_frames)}
                for future in concurrent.futures.as_completed(future_results):
                    _ = future.result()
                    pbar.update(1)
        if DEBUG:
            break

    # cube_sz = [3, 7, 7]

    # def create_cube(i_time, ind_neuron, neuron):
    #     # Inner loop: one time and one neuron
    #     this_df = df[neuron]
    #     # FLIP XY
    #     z, y, x = [int(this_df['z'][i_time]), int(this_df['x'][i_time]), int(this_df['y'][i_time])]
    #     # Instead do a cube (just for visualization)
    #     z, x, y = get_crop_coords3d([z, x, y], cube_sz, chunk_sz)
    #     new_masks[i_time, z[0]:z[-1]+1, x[0]:x[-1]+1, y[0]:y[-1]+1] = ind_neuron
    #
    # def process_single_volume(i_time):
    #     # Outer loop: process one full volume
    #     for i_neuron, neuron in enumerate(neuron_names):
    #         create_cube(i_time, i_neuron, neuron)
    #
    # # Instead do a process pool that finishes one file at a time
    # # NOTE: blosc can hang when doing the multiprocesssing :(
    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
    #     with tqdm(total=num_frames) as progress:
    #         futures = []
    #
    #         for t in range(num_frames):
    #             future = pool.submit(process_single_volume, t)
    #             future.add_done_callback(lambda p: progress.update())
    #             futures.append(future)
    #
    #         results = []
    #         for future in futures:
    #             result = future.result()
    #             results.append(result)
    #
    # # Reset to automatic detection
    # blosc.use_threads = None


def all_matches_to_lookup_tables(all_matches: dict) -> dict:
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
        try:
            # TODO: are the matches always the same length?
            dlc_ind = np.array(match)[:, 0].astype(int)
            seg_ind = np.array(match)[:, 1].astype(int)
            lut = np.zeros(1000, dtype=int)  # TODO: Should be more than the maximum local index
            lut[seg_ind] = dlc_ind  # Raw indices of the lut should match the local index
            # if np.max(seg_ind) > 1000:
            #     raise ValueError("Lookup-table size is too small; increase this (in code) or fix it!")
        except IndexError:
            # Some volumes may be empty
            pass
        all_lut[i_volume] = lut
    return all_lut


def reindex_segmentation_only_training_data(cfg: modular_project_config,
                                            segment_cfg: config_file_with_project_context,
                                            tracking_cfg: config_file_with_project_context,
                                            DEBUG=False):
    """
    Using tracklets and full segmentation, produces a small video (zarr) with neurons colored by track
    """

    num_frames = cfg.config['dataset_params']['num_frames']

    # Get ALL matches to the segmentation, then subset
    with safe_cd(cfg.project_dir):
        # TODO: not hardcoded
        fname = os.path.join('2-training_data', 'raw', 'clust_df_dat.pickle')
        df = pd.read_pickle(fname)

        # Get the frames chosen as training data, or recalculate
        which_frames = get_or_recalculate_which_frames(DEBUG, df, num_frames, tracking_cfg)
        # logging.log(f"Which frames to use for training data: {which_frames}")

        # Build a sub-df with only the relevant neurons; all slices
        # Todo: connect up to actually tracked z slices?
        subset_opt = {'which_z': None,
                      'max_z_dist': None,
                      'verbose': 1}
        subset_df = build_subset_df_from_tracklets(df, which_frames, **subset_opt)

        fname = segment_cfg.config['output_metadata']
        with open(fname, 'rb') as f:
            segmentation_metadata = pickle.load(f)

        fname = segment_cfg.config['output_masks']
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
    out_fname = os.path.join(cfg.project_dir, '2-training_data', 'reindexed_masks.zarr')
    new_masks = zarr.open_like(masks, path=out_fname, shape=new_sz)

    # Reindex; this automatically writes to disk
    for i, (i_volume, lut) in tqdm(enumerate(all_lut.items())):
        new_masks[i, ...] = lut[masks[i_volume, ...]]

    # Automatically saves
