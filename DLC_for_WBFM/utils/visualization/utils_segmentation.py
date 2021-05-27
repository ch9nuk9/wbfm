import os

from DLC_for_WBFM.utils.projects.utils_project import load_config
import pandas as pd
import numpy as np
import zarr


def reindex_segmentation(project_path):
    """
    Reindexes segmentation, which originally has arbitrary numbers, to reflect tracking
    """
    cfg = load_config(project_path)

    # Get original segmentation
    seg_cfg = load_config(cfg['subfolder_configs']['segmentation'])
    seg_fname = seg_cfg['output']['masks']
    seg_masks = zarr.open(seg_fname)

    out_fname = os.path.join("4-trackes", "reindexed_masks.zarr")
    new_masks = zarr.zeros_like(seg_masks, path=out_fname)

    # Get tracking (dataframe) with neuron names
    trace_cfg = load_config(cfg['subfolder_configs']['traces'])
    matches_fname = trace_cfg['traces']['all_matches']
    all_matches = pd.read_hdf(matches_fname)
    # Format: dict with i_volume -> Nx3 array of [local_ind, global_ind, confidence] triplets

    # Convert dataframe to lookup tables, per volume
    # Note: if not all neurons are in the dataframe, then they are set to 0
    all_lut = {}
    for i_volume, match in all_matches.items():
        lut = np.zeros_like(match[:, 0])
        lut[match[:, 0]] = match[:, 1]  # Raw indices of the lut should match the local index
        all_lut[i_volume] = lut

    # Apply lookup tables to each volume
    # Also see link for ways to speed this up:
    # https://stackoverflow.com/questions/14448763/is-there-a-convenient-way-to-apply-a-lookup-table-to-a-large-array-in-numpy
    for i_volume, lut in all_lut.items():
        new_masks[i_volume, ...] = lut[seg_masks[i_volume, ...]]

