import concurrent.futures
import os
from pathlib import Path

from tqdm.auto import tqdm

from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
import pandas as pd
import numpy as np
import zarr


def reindex_segmentation(project_path, DEBUG=False):
    """
    Reindexes segmentation, which originally has arbitrary numbers, to reflect tracking
    """
    cfg = load_config(project_path)

    with safe_cd(Path(project_path).parent):
        # Get original segmentation
        seg_cfg = load_config(cfg['subfolder_configs']['segmentation'])
        seg_fname = seg_cfg['output']['masks']
        seg_masks = zarr.open(seg_fname)

        out_fname = os.path.join("4-traces", "reindexed_masks.zarr")
        print(f"Saving masks at {out_fname}")
        new_masks = zarr.open_like(seg_masks, path=out_fname)

        # Get tracking (dataframe) with neuron names
        trace_cfg = load_config(cfg['subfolder_configs']['traces'])
        matches_fname = Path(trace_cfg['all_matches'])
        all_matches = pd.read_pickle(matches_fname)
        # Format: dict with i_volume -> Nx3 array of [dlc_ind, segmentation_ind, confidence] triplets

    # Convert dataframe to lookup tables, per volume
    # Note: if not all neurons are in the dataframe, then they are set to 0
    all_lut = {}
    for i_volume, match in tqdm(all_matches.items()):
        dlc_ind = match[:, 0].astype(int)
        seg_ind = match[:, 1].astype(int)
        lut = np.zeros(1000, dtype=int)  # TODO: Should be more than the maximum local index
        # TODO: are the matches always the same length?
        lut[seg_ind] = dlc_ind  # Raw indices of the lut should match the local index
        all_lut[i_volume] = lut

    # err
    # Apply lookup tables to each volume
    # Also see link for ways to speed this up:
    # https://stackoverflow.com/questions/14448763/is-there-a-convenient-way-to-apply-a-lookup-table-to-a-large-array-in-numpy
    # for i_volume, lut in tqdm(all_lut.items()):
    #     new_masks[i_volume, ...] = lut[seg_masks[i_volume, ...]]
    #     if DEBUG:
    #         print("DEBUG mode; quitting after first volume")
    #         break

    def parallel_func(i):
        lut = all_lut[i]
        new_masks[i, ...] = lut[seg_masks[i, ...]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # executor.map(parallel_func, range(len(all_lut)))
        future_results = {executor.submit(parallel_func, i): i for i in range(len(all_lut))}
        for future in concurrent.futures.as_completed(future_results):
            # _ = future_results[future]
            _ = future.result()
