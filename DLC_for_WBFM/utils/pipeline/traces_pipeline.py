import concurrent
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_icp_matches
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config, config_file_with_project_context
from DLC_for_WBFM.utils.projects.utils_project import edit_config, safe_cd
from DLC_for_WBFM.utils.visualization.utils_segmentation import reindex_segmentation_using_config


def get_traces_from_3d_tracks_using_config(segment_cfg: config_file_with_project_context,
                                           track_cfg: config_file_with_project_context,
                                           traces_cfg: config_file_with_project_context,
                                           project_cfg: modular_project_config,
                                           DEBUG: bool = False) -> None:
    """
    Connect the 3d traces to previously segmented masks

    Get both red and green traces for each neuron
    """
    dlc_tracks, green_fname, red_fname, max_dist, num_frames, params_start_volume, segmentation_metadata, z_to_xy_ratio = _unpack_configs_for_traces(
        project_cfg, segment_cfg, track_cfg)

    green_video = zarr.open(green_fname)
    red_video = zarr.open(red_fname)

    # Match -> Reindex -> Get traces
    old_dlc_names = list(dlc_tracks.columns.levels[0])

    def _get_dlc_zxy(t, dlc_tracks=dlc_tracks):
        all_dlc_zxy = np.zeros((len(old_dlc_names), 3))
        coords = ['z', 'y', 'x']
        for i, name in enumerate(old_dlc_names):
            all_dlc_zxy[i, :] = np.asarray(dlc_tracks[name][coords].loc[t])
        return all_dlc_zxy

    # Main loop: Match segmentations to tracks
    # Also: get connected red brightness and mask
    # Initialize multi-index dataframe for data
    frame_list = list(range(params_start_volume, num_frames + params_start_volume))
    all_matches = defaultdict(list)  # key = i_vol; val = Nx3-element list
    print("Matching segmentation and DLC tracking...")
    if DEBUG:
        frame_list = frame_list[:2]  # Shorten (to avoid break)
    calculate_segmentation_and_dlc_matches(_get_dlc_zxy, all_matches, frame_list, max_dist,
                                           segmentation_metadata, z_to_xy_ratio, DEBUG=DEBUG)

    relative_fname = traces_cfg.config['all_matches']
    project_cfg.save_in_local_project(all_matches, relative_fname)

    print("Reindexing masks using matches...")
    # Reads matches from disk, and then saves the masks
    reindex_segmentation_using_config(traces_cfg, segment_cfg, project_cfg)

    print("Extracting red and green traces using reindexed masks...")
    # Reads masks from disk, and writes traces
    fname = traces_cfg.resolve_relative_path_from_config('reindexed_masks')
    with safe_cd(project_cfg.project_dir):
        reindexed_masks = zarr.open(fname)

    # New: rename neurons to be same as segmentation indices
    new_neuron_names = [f"neuron{i + 1}" for i in range(len(old_dlc_names))]
    dlc_name_mapping = dict(zip(old_dlc_names, new_neuron_names))

    def _get_dlc_zxy_one_neuron(t, new_name):
        old_name = dlc_name_mapping[new_name]
        coords = ['z', 'y', 'x']
        all_dlc_zxy = np.asarray(dlc_tracks[old_name][coords].loc[t])
        return all_dlc_zxy

    def parallel_func(i_and_name):
        i, new_name = i_and_name
        return calc_trace_from_mask_one_neuron(_get_dlc_zxy_one_neuron, frame_list, green_video, red_video,
                                               i, new_name,
                                               params_start_volume,
                                               reindexed_masks)

    with tqdm(total=len(new_neuron_names)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(parallel_func, i): i for i in enumerate(new_neuron_names)}
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    all_green_dfs = [r[0] for r in results]
    all_red_dfs = [r[1] for r in results]

    df_green = pd.concat(all_green_dfs, axis=1)
    df_red = pd.concat(all_red_dfs, axis=1)

    if DEBUG:
        print("Single pass-through successful")

    _save_traces_as_hdf_and_update_configs(new_neuron_names, df_green, df_red, traces_cfg)


def calc_trace_from_mask_one_neuron(_get_dlc_zxy_one_neuron, frame_list, green_video, red_video,
                                    i, new_name, params_start_volume,
                                    reindexed_masks):
    all_green_dfs_one_neuron = []
    all_red_dfs_one_neuron = []
    i_mask_ind = i + 1
    for i_volume in tqdm(frame_list, leave=False):
        this_zxy_dlc = _get_dlc_zxy_one_neuron(i_volume, new_name)
        # Prepare mask (segmentation)
        i_mask = i_volume - params_start_volume
        this_mask_volume = reindexed_masks[i_mask, ...]
        this_mask_neuron = (this_mask_volume == i_mask_ind)

        # Green then red
        this_green_volume = green_video[i_volume, ...]
        df_green_one_frame = extract_traces_using_reindexed_masks(new_name, this_zxy_dlc,
                                                                  i_mask_ind, i_volume,
                                                                  this_green_volume, this_mask_neuron)
        this_red_volume = red_video[i_volume, ...]
        df_red_one_frame = extract_traces_using_reindexed_masks(new_name, this_zxy_dlc,
                                                                i_mask_ind, i_volume,
                                                                this_red_volume, this_mask_neuron)
        all_green_dfs_one_neuron.append(df_green_one_frame)
        all_red_dfs_one_neuron.append(df_red_one_frame)
    df_green_one_neuron = pd.concat(all_green_dfs_one_neuron, axis=0)
    df_red_one_neuron = pd.concat(all_red_dfs_one_neuron, axis=0)
    return df_green_one_neuron, df_red_one_neuron


def _save_traces_as_hdf_and_update_configs(new_neuron_names: list,
                                           df_green: pd.DataFrame,
                                           df_red: pd.DataFrame,
                                           traces_cfg: config_file_with_project_context) -> None:
    # Save traces (red and green) and neuron names
    # csv doesn't work well when some entries are lists
    red_fname = Path('4-traces').joinpath('red_traces.h5')
    df_red.to_hdf(str(red_fname), "df_with_missing")

    green_fname = Path('4-traces').joinpath('green_traces.h5')
    df_green.to_hdf(str(green_fname), "df_with_missing")

    # Save the output filenames
    traces_cfg.config['traces']['green'] = str(green_fname)
    traces_cfg.config['traces']['red'] = str(red_fname)
    traces_cfg.config['traces']['neuron_names'] = new_neuron_names
    traces_cfg.update_on_disk()
    # edit_config(traces_cfg.config['self_path'], traces_cfg)


def calculate_segmentation_and_dlc_matches(_get_dlc_zxy: Callable,
                                           all_matches: defaultdict,
                                           frame_list: list,
                                           max_dist: float,
                                           segmentation_metadata: Dict[int, pd.DataFrame],
                                           z_to_xy_ratio: float, DEBUG: bool = False) -> None:
    """

    Parameters
    ----------
    _get_dlc_zxy
    all_matches
    all_neuron_names
    frame_list
    max_dist
    red_dat
    segmentation_metadata
    z_to_xy_ratio

    Returns
    -------
    None
    """
    for i_volume in tqdm(frame_list):
        # Get DLC point cloud
        # NOTE: This dataframe starts at 0, not start_volume
        zxy0 = _get_dlc_zxy(i_volume)
        zxy0[:, 0] *= z_to_xy_ratio
        # REVIEW: Get segmentation point cloud
        seg_zxy = segmentation_metadata[i_volume]['centroids']
        seg_zxy = [np.asarray(row) for row in seg_zxy]
        if len(seg_zxy) == 0:
            continue
        zxy1 = np.array(seg_zxy)
        zxy1[:, 0] *= z_to_xy_ratio
        # Get matches
        out = calc_icp_matches(zxy0, zxy1, max_dist=max_dist)
        # TODO: the distance function doesn't produce the correct reindexed segmentations
        # out = calc_bipartite_from_distance(zxy0, zxy1, max_dist=max_dist)
        matches, conf, _ = out

        def seg_array_to_ind(i):
            # The seg_zxy array has the 0th row corresponding to segmentation mask label 1
            return i + 1

        def dlc_array_to_ind(i):
            # _get_dlc_zxy has the 0th row corresponding to neuron label 1
            return i + 1

        # Save
        # NOTE: need to offset by 1, because the background is 0
        all_matches[i_volume] = np.array(
            [[seg_array_to_ind(m[0]), dlc_array_to_ind(m[1]), c[0]] for m, c in zip(matches, conf)]
        )


def _unpack_configs_for_traces(project_cfg, segment_cfg, track_cfg):
    # Settings
    max_dist = track_cfg.config['final_3d_tracks']['max_dist_to_segmentation']
    params_start_volume = project_cfg.config['dataset_params']['start_volume']
    num_frames = project_cfg.config['dataset_params']['num_frames']
    # Get previous annotations
    segmentation_fname = segment_cfg.resolve_relative_path_from_config('output_metadata')
    with open(segmentation_fname, 'rb') as f:
        segmentation_metadata = pickle.load(f)
    dlc_fname = track_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
    z_to_xy_ratio = project_cfg.config['dataset_params']['z_to_xy_ratio']
    green_fname = project_cfg.config['preprocessed_green']
    red_fname = project_cfg.config['preprocessed_red']

    dlc_tracks: pd.DataFrame = pd.read_hdf(dlc_fname)

    return dlc_tracks, green_fname, red_fname, max_dist, num_frames, params_start_volume, segmentation_metadata, z_to_xy_ratio


def _initialize_dataframe(all_neuron_names: List[str], frame_list: List[int]) -> pd.DataFrame:
    m_index = _get_multiindex(all_neuron_names)
    sz = (len(frame_list), len(m_index))
    empty_dat = np.empty(sz)
    empty_dat[:] = np.nan
    df_red = pd.DataFrame(empty_dat,
                          columns=m_index,
                          index=frame_list)
    for name in all_neuron_names:
        # Allow saving numpy arrays in the column
        df_red[(name, 'all_values')] = df_red[(name, 'all_values')].astype('object')
    return df_red


def _get_multiindex(all_neuron_names: List[str]) -> pd.MultiIndex:
    save_names = ['brightness', 'volume',
                  'all_values',
                  'i_reindexed_segmentation',
                  'z_dlc', 'x_dlc', 'y_dlc',
                  'match_confidence']
    m_index = pd.MultiIndex.from_product([all_neuron_names,
                                          save_names],
                                         names=['neurons', 'data'])
    return m_index


def extract_traces_using_reindexed_masks(d_name: str, zxy_dlc: np.ndarray,
                                         i_mask: int, i_volume: int,
                                         video_volume: np.ndarray,
                                         this_mask_neuron: np.ndarray, confidence: float = 0.0) -> pd.DataFrame:
    i = i_volume
    # Get brightness from green volume and mask
    # Use reindexed mask instead of original index mask
    volume = np.count_nonzero(this_mask_neuron)
    if volume > 0:
        all_values = video_volume[this_mask_neuron]
        brightness = np.sum(all_values)
    else:
        brightness = np.nan
        volume = np.nan
        all_values = []

    # Save in dataframe
    df_as_dict = {
        (d_name, 'brightness'): brightness,
        (d_name, 'volume'): volume,
        (d_name, 'all_values'): [all_values],
        (d_name, 'i_reindexed_segmentation'): i_mask,
        (d_name, 'z_dlc'): zxy_dlc[0],
        (d_name, 'x_dlc'): zxy_dlc[1],
        (d_name, 'y_dlc'): zxy_dlc[2],
        (d_name, 'match_confidence'): confidence,
    }

    df = pd.DataFrame(df_as_dict, index=[i])

    return df


def OLD_extract_traces_using_reindexed_masks(d_name: List[str], all_zxy_dlc: np.ndarray,
                                             df: pd.DataFrame, i_mask: int, i_volume: int,
                                             is_mirrored: bool, video_volume: np.ndarray,
                                             this_mask_volume: np.ndarray, confidence: float = 0.0):
    # For conversion between lists
    # i_dlc, i_seg = int(i_dlc), int(i_seg)
    # s_name = int(all_seg_names[i_seg])
    i = i_volume
    # Get brightness from green volume and mask
    # this_mask_neuron = (this_mask_volume == s_name)
    # Use reindexed mask instead of original index mask
    this_mask_neuron = (this_mask_volume == i_mask)
    if is_mirrored:
        this_mask_neuron = np.flip(this_mask_neuron, axis=2)
    volume = np.count_nonzero(this_mask_neuron)
    all_values = video_volume[this_mask_neuron]
    brightness = np.sum(all_values)
    # Save in dataframe
    df[(d_name, 'brightness')].loc[i] = brightness
    df[(d_name, 'volume')].loc[i] = volume
    # df[(d_name, 'centroid_ind')].loc[i] = s_name
    df[(d_name, 'all_values')].loc[i] = all_values
    df[(d_name, 'i_reindexed_segmentation')].loc[i] = i_mask
    # zxy_seg = mdat['centroids'][s_name]
    zxy_dlc = all_zxy_dlc[i_mask - 1]
    _save_locations_in_df(d_name, df, i, zxy_dlc, confidence)


def _save_locations_in_df(d_name, df, i, zxy_dlc, conf):
    df[(d_name, 'z_dlc')].loc[i] = zxy_dlc[0]
    df[(d_name, 'x_dlc')].loc[i] = zxy_dlc[1]
    df[(d_name, 'y_dlc')].loc[i] = zxy_dlc[2]
    df[(d_name, 'match_confidence')].loc[i] = conf


def rebuild_pixel_values(frame_df, which_neuron):
    # I accidentally didn't save the full histogram... so I remake it
    counts, edges = frame_df.loc[which_neuron, 'pixel_counts'], frame_df.loc[which_neuron, 'pixel_values']
    pixel_vals = np.repeat(edges, counts)
    return pixel_vals
