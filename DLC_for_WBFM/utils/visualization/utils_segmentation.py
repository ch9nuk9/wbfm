import concurrent.futures
import os
import pickle
from pathlib import Path

from tqdm.auto import tqdm

from DLC_for_WBFM.utils.feature_detection.visualize_using_dlc import build_subset_df
from DLC_for_WBFM.utils.pipeline.dlc_pipeline import _get_frames_for_dlc_training
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
import pandas as pd
import numpy as np
import zarr


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
    all_lut = {}
    for i_volume, match in all_matches.items():
        dlc_ind = match[:, 0].astype(int)
        seg_ind = match[:, 1].astype(int)
        lut = np.zeros(1000, dtype=int)  # TODO: Should be more than the maximum local index
        # TODO: are the matches always the same length?
        lut[seg_ind] = dlc_ind  # Raw indices of the lut should match the local index
        all_lut[i_volume] = lut
    return all_lut


def reindex_segmentation_only_training_data(this_config, DEBUG=False):
    """
    Using tracklets and full segmentation, produces a small video (zarr) with neurons colored by track
    """


    # Get ALL matches to the segmentation, then subset
    # Also get segmentation metadata
    with safe_cd(this_config['project_path']):
        fname = os.path.join('2-training_data', 'raw', 'clust_df_dat.pickle')
        df = pd.read_pickle(fname)

        # Get the frames chosen as training data
        try:
            which_frames = this_config['track_cfg']['training_data_3d']['which_frames']
        except KeyError:
            which_frames, _ = _get_frames_for_dlc_training(DEBUG, this_config, df)

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

    # Initialize
    new_sz = list(masks.shape)
    new_sz[0] = len(which_frames)
    out_fname = os.path.join(this_config['project_path'], '2-training_data', 'reindexed_segmentation.zarr')
    new_masks = zarr.open_like(masks, path=out_fname, shape=new_sz)

    # Reindex; this automatically writes to disk
    for i, (i_volume, lut) in tqdm(enumerate(all_lut.items())):
        new_masks[i, ...] = lut[masks[i_volume, ...]]

    # Note: automatically saves
