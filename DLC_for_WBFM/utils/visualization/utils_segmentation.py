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

    # Get tracking (dataframe) with neuron names
    track_cfg = load_config(cfg['subfolder_configs']['tracking'])
    seg_fname = seg_cfg['output']['masks']


    # Convert dataframe to lookup tables, per volume
