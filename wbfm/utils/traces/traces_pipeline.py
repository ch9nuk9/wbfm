import os
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import zarr
from tqdm.auto import tqdm

from wbfm.utils.neuron_matching.utils_matching import calc_nearest_neighbor_matches
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import SubfolderConfigFile
from wbfm.utils.tracklets.training_data_from_tracklets import build_subset_df_from_tracklets


def extract_traces_of_training_data_from_config(project_cfg: SubfolderConfigFile,
                                                training_cfg: SubfolderConfigFile,
                                                name_mode='tracklet'):
    """Principally used for positions, but the rest could be useful for quality control"""
    project_data = ProjectData.load_final_project_data_from_config(project_cfg, to_load_tracklets=True)
    df = project_data.df_all_tracklets
    which_frames = project_data.which_training_frames

    df_train = build_subset_df_from_tracklets(df, which_frames)
    df_train = df_train.dropna().reset_index(drop=True)

    # Note that the training config works with the same function as step 4c for getting the local masks
    fname = os.path.join("2-training_data", "training_data_tracks.h5")
    if hasattr(df_train, 'sparse'):
        df_train = df_train.sparse.to_dense()
    training_cfg.save_data_in_local_project(df_train, fname, also_save_csv=True)


def _save_traces_as_hdf_and_update_configs(final_neuron_names: list,
                                           df_green: pd.DataFrame,
                                           df_red: pd.DataFrame,
                                           traces_cfg: SubfolderConfigFile) -> None:
    # Save traces (red and green) and neuron names
    # csv doesn't work well when some entries are lists
    red_fname = Path('4-traces').joinpath('red_traces.h5')
    traces_cfg.save_data_in_local_project(df_red, str(red_fname))

    green_fname = Path('4-traces').joinpath('green_traces.h5')
    traces_cfg.save_data_in_local_project(df_green, str(green_fname))

    # Save the output filenames
    traces_cfg.config['traces']['green'] = str(green_fname)
    traces_cfg.config['traces']['red'] = str(red_fname)
    traces_cfg.config['traces']['neuron_names'] = final_neuron_names
    traces_cfg.update_self_on_disk()


def match_segmentation_and_tracks(_get_zxy_from_pandas: Callable,
                                  all_matches: defaultdict,
                                  frame_list: list,
                                  max_dist: float,
                                  project_data: ProjectData,
                                  DEBUG: bool = False) -> None:
    """

    Parameters
    ----------
    _get_zxy_from_pandas
    all_matches
    frame_list
    max_dist
    project_data

    Returns
    -------
    None
    """
    for i_volume in tqdm(frame_list):
        # Get tracking point cloud
        # NOTE: This dataframe starts at 0, not start_volume
        zxy0 = _get_zxy_from_pandas(i_volume)
        # TODO: use physical units and align between z and xy
        zxy1 = project_data.get_centroids_as_numpy(i_volume)
        if len(zxy1) == 0 or len(zxy0) == 0:
            continue
        # Get matches
        matches, conf, = calc_nearest_neighbor_matches(zxy0, zxy1, max_dist=max_dist)

        def seg_array_to_mask_ind(i):
            # The seg_zxy array has the 0th row corresponding to segmentation mask label 1
            # BUT can also skip rows and might generally be non-monotonic
            return project_data.segmentation_metadata.i_in_array_to_mask_index(i_volume, i)

        def dlc_array_to_ind(i):
            # the 0th index corresponds to neuron_001, and should finally be label 1
            return i + 1

        # Save
        all_matches[i_volume] = np.array(
            [[dlc_array_to_ind(m[0]), seg_array_to_mask_ind(m[1]), c] for m, c in zip(matches, conf)]
        )


def _unpack_configs_for_traces(project_cfg, track_cfg):
    # Settings
    max_dist = track_cfg.config['final_3d_tracks']['max_dist_to_segmentation']
    params_start_volume = project_cfg.config['dataset_params']['start_volume']
    num_frames = project_cfg.config['dataset_params']['num_frames']

    return max_dist, num_frames, params_start_volume


def _unpack_configs_for_extraction(project_cfg, traces_cfg):
    # Settings
    params_start_volume = project_cfg.config['dataset_params']['start_volume']
    num_frames = project_cfg.config['dataset_params']['num_frames']

    frame_list = list(range(params_start_volume, num_frames + params_start_volume))

    coords = ['z', 'x', 'y']
    fname = traces_cfg.resolve_relative_path_from_config('reindexed_masks')
    reindexed_masks = zarr.open(fname)

    return coords, reindexed_masks, frame_list, params_start_volume
