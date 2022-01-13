import concurrent
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
import zarr
from skimage import measure
from tqdm import tqdm

from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_icp_matches, calc_bipartite_from_distance, \
    calc_nearest_neighbor_matches
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig, SubfolderConfigFile, pickle_load_binary
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name_neuron


def get_traces_from_3d_tracks_using_config(segment_cfg: SubfolderConfigFile,
                                           track_cfg: SubfolderConfigFile,
                                           traces_cfg: SubfolderConfigFile,
                                           project_cfg: ModularProjectConfig,
                                           DEBUG: bool = False) -> None:
    """
    Connect the 3d traces to previously segmented masks

    Get both red and green traces for each neuron
    """
    dlc_tracks, green_fname, red_fname, max_dist, num_frames, params_start_volume, z_to_xy_ratio = _unpack_configs_for_traces(
        project_cfg, segment_cfg, track_cfg)

    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    # Match -> Reindex raw segmentation -> Get traces
    final_neuron_names = list(dlc_tracks.columns.levels[0])
    assert 'neuron0' not in final_neuron_names, "Neuron0 found; 0 is reserved for background... check original " \
                                                "dataframe generation and indexing"

    coords = ['z', 'x', 'y']

    def _get_dlc_zxy(t):
        all_dlc_zxy = np.zeros((len(final_neuron_names), 3))
        for i, name in enumerate(final_neuron_names):
            all_dlc_zxy[i, :] = np.asarray(dlc_tracks[name][coords].loc[t])
        return all_dlc_zxy

    # Main loop: Match segmentations to tracks
    # Also: get connected red brightness and mask
    # Initialize multi-index dataframe for data
    # TODO: Why is this one frame too short?
    frame_list = list(range(params_start_volume, num_frames + params_start_volume - 1))
    all_matches = defaultdict(list)  # key = i_vol; val = Nx3-element list
    logging.info("Matching segmentation and tracked positions...")
    if DEBUG:
        frame_list = frame_list[:2]  # Shorten (to avoid break)
    calculate_segmentation_and_dlc_matches(_get_dlc_zxy, all_matches, frame_list, max_dist,
                                           project_data, z_to_xy_ratio, DEBUG=DEBUG)

    relative_fname = traces_cfg.config['all_matches']
    project_cfg.pickle_in_local_project(all_matches, relative_fname)


def extract_traces_using_config(project_cfg: SubfolderConfigFile,
                                traces_cfg: SubfolderConfigFile,
                                DEBUG):
    """
    Final step that loops through original data and extracts traces using labeled masks
    """
    coords, reindexed_masks, frame_list, params_start_volume = \
        _unpack_configs_for_extraction(project_cfg, traces_cfg)
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    red_all_neurons, green_all_neurons = region_props_all_volumes(
        reindexed_masks,
        project_data.red_data,
        project_data.green_data,
        frame_list,
        params_start_volume
    )

    df_green, df_red = _convert_nested_dict_to_dataframe(coords, frame_list, green_all_neurons, red_all_neurons)

    # TODO: make sure these are strings
    final_neuron_names = list(df_red.columns.levels[0])

    _save_traces_as_hdf_and_update_configs(final_neuron_names, df_green, df_red, traces_cfg)


def _convert_nested_dict_to_dataframe(coords, frame_list, green_all_neurons, red_all_neurons):
    # Convert nested dict of volumes to final dataframes
    sz_one_neuron = len(frame_list)
    i_start = min(list(red_all_neurons.keys()))
    tmp_red = defaultdict(lambda: np.zeros(sz_one_neuron))
    tmp_green = defaultdict(lambda: np.zeros(sz_one_neuron))
    volume_indices = list(red_all_neurons.keys())
    volume_indices.sort()
    for i_vol in volume_indices:
        red_props, green_props = red_all_neurons[i_vol], green_all_neurons[i_vol]
        i_vol_numpy = i_vol - i_start
        for key in red_props.keys():
            if 'weighted_centroid' in key:
                # Later formatting expects this to be split, i.e. z x y are separate columns
                for i, c in enumerate(coords):
                    new_key = (key[0], c)
                    tmp_red[new_key][i_vol_numpy] = red_props[key][i]
                    tmp_green[new_key][i_vol_numpy] = green_props[key][i]
            else:
                tmp_red[key][i_vol_numpy] = red_props[key]
                tmp_green[key][i_vol_numpy] = green_props[key]
    df_red = pd.DataFrame(tmp_red, index=volume_indices)
    df_green = pd.DataFrame(tmp_green, index=volume_indices)
    return df_green, df_red


def make_mask2final_mapping(all_matches: dict):
    mask2final_name_per_volume = {}

    for i_volume, match in all_matches.items():
        match = np.array(match)
        if len(match) == 0:
            # Rarely, there are just no matches
            continue
        dlc_ind = match[:, 0].astype(int)
        seg_ind = match[:, 1].astype(int)
        mask2final_name_per_volume[i_volume] = dict(zip(dlc_ind, seg_ind))

    return mask2final_name_per_volume


def region_props_all_volumes(reindexed_masks, red_video, green_video,
                             frame_list,
                             params_start_volume):
    """

    Parameters
    ----------
    reindexed_masks
    red_video
    green_video
    frame_list
    params_start_volume

    Returns
    -------
    Two nested dictionaries: red_all_neurons, green_all_neurons
        Outer keys are volume indices, and inner keys are neuron id + property name (tuple)
    """
    red_all_neurons = {}
    green_all_neurons = {}

    # options = dict(
    #     green_all_neurons=green_all_neurons,
    #     green_video=green_video,
    #     mask2final_name_per_volume=mask2final_name_per_volume,
    #     params_start_volume=params_start_volume,
    #     red_all_neurons=red_all_neurons,
    #     red_video=red_video,
    #     reindexed_masks=reindexed_masks
    # )

    # def _parallel_func(i_volume, green_all_neurons, green_video, mask2final_name_per_volume, params_start_volume,
    #                    red_all_neurons, red_video, reindexed_masks):
    def _parallel_func(i_volume):
        i_mask = i_volume - params_start_volume
        this_mask_volume = reindexed_masks[i_mask, ...]
        this_green_volume = green_video[i_volume, ...]
        this_red_volume = red_video[i_volume, ...]
        # mask2final_name = mask2final_name_per_volume[i_volume]
        red_one_vol, green_one_vol = region_props_one_volume(
            this_mask_volume,
            this_red_volume,
            this_green_volume
        )
        red_all_neurons[i_volume] = red_one_vol
        green_all_neurons[i_volume] = green_one_vol

    with tqdm(total=len(frame_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(_parallel_func, i): i for i in frame_list}
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()
                pbar.update(1)

    return red_all_neurons, green_all_neurons


def region_props_one_volume(this_mask_volume,
                            this_red_volume,
                            this_green_volume):
    """

    Parameters
    ----------
    this_green_volume
    this_mask_volume
    this_red_volume

    Returns
    -------
    Two dictionaries with keys that are 2-tuples (i_final_neuron, property_name) with properties:
        props_to_save = ['area', 'weighted_centroid', 'image_intensity', 'label']

    """
    props_to_save = ['area', 'weighted_centroid', 'intensity_image', 'label']
    red_neurons_one_volume = regionprops_one_volume_one_channel(this_mask_volume, this_red_volume, props_to_save)
    green_neurons_one_volume = regionprops_one_volume_one_channel(this_mask_volume, this_green_volume, props_to_save)

    # red_neurons_one_volume = {}
    # green_neurons_one_volume = {}
    #
    # # Green then red
    # green_props = measure.regionprops(this_mask_volume, intensity_image=this_green_volume)
    # red_props = measure.regionprops(this_mask_volume, intensity_image=this_red_volume)
    #
    # for this_red, this_green in zip(red_props, green_props):
    #     seg_index = this_red['label']
    #     # final_index = mask2final_name[seg_index]
    #     key_base = (int2name_neuron(seg_index),)
    #
    #     for this_prop in props_to_save:
    #         key = key_base + (this_prop,)
    #         if this_prop == 'intensity_image':
    #             red_neurons_one_volume[key] = np.sum(this_red[this_prop])
    #             green_neurons_one_volume[key] = np.sum(this_green[this_prop])
    #         else:
    #             red_neurons_one_volume[key] = this_red[this_prop]
    #             green_neurons_one_volume[key] = this_green[this_prop]

    return red_neurons_one_volume, green_neurons_one_volume


def regionprops_one_volume_one_channel(mask, data, props_to_save):
    neurons_one_volume = {}
    props = measure.regionprops(mask, intensity_image=data)

    for this_neuron in props:
        seg_index = this_neuron['label']
        # final_index = mask2final_name[seg_index]
        key_base = (int2name_neuron(seg_index),)

        for this_prop in props_to_save:
            key = key_base + (this_prop,)
            if this_prop == 'intensity_image':
                neurons_one_volume[key] = np.sum(this_neuron[this_prop])
            else:
                neurons_one_volume[key] = this_neuron[this_prop]

    return neurons_one_volume



def _pool_parallel_func(i_and_name, options):
    i, new_name = i_and_name
    return _pool_calc_trace_from_mask_one_neuron(i=i, new_name=new_name, **options)


def _pool_get_dlc_zxy_one_neuron(t, new_name, dlc_name_mapping, dlc_tracks):
    old_name = dlc_name_mapping[new_name]
    coords = ['z', 'x', 'y']
    all_dlc_zxy = np.asarray(dlc_tracks[old_name][coords].loc[t])
    return all_dlc_zxy


def _pool_calc_trace_from_mask_one_neuron(frame_list, green_video, red_video,
                                     i, new_name, params_start_volume,
                                     reindexed_masks,
                                     dlc_name_mapping,
                                     dlc_tracks):
    all_green_dfs_one_neuron = []
    all_red_dfs_one_neuron = []
    i_mask_ind = i + 1
    for i_volume in tqdm(frame_list, leave=False):
        this_zxy_dlc = _pool_get_dlc_zxy_one_neuron(i_volume, new_name, dlc_name_mapping, dlc_tracks)
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


def _save_traces_as_hdf_and_update_configs(final_neuron_names: list,
                                           df_green: pd.DataFrame,
                                           df_red: pd.DataFrame,
                                           traces_cfg: SubfolderConfigFile) -> None:
    # Save traces (red and green) and neuron names
    # csv doesn't work well when some entries are lists
    red_fname = Path('4-traces').joinpath('red_traces.h5')
    df_red.to_hdf(str(red_fname), "df_with_missing")

    green_fname = Path('4-traces').joinpath('green_traces.h5')
    df_green.to_hdf(str(green_fname), "df_with_missing")

    # Save the output filenames
    traces_cfg.config['traces']['green'] = str(green_fname)
    traces_cfg.config['traces']['red'] = str(red_fname)
    traces_cfg.config['traces']['neuron_names'] = final_neuron_names
    traces_cfg.update_on_disk()
    # edit_config(traces_cfg.config['self_path'], traces_cfg)


def calculate_segmentation_and_dlc_matches(_get_dlc_zxy: Callable,
                                           all_matches: defaultdict,
                                           frame_list: list,
                                           max_dist: float,
                                           project_data: ProjectData,
                                           z_to_xy_ratio: float, DEBUG: bool = False) -> None:
    """

    Parameters
    ----------
    _get_dlc_zxy
    all_matches
    frame_list
    max_dist
    project_data
    z_to_xy_ratio

    Returns
    -------
    None
    """
    for i_volume in tqdm(frame_list):
        # Get DLC point cloud
        # NOTE: This dataframe starts at 0, not start_volume
        zxy0 = _get_dlc_zxy(i_volume)
        # TODO: use physical units and align between z and xy
        # zxy0[:, 0] *= z_to_xy_ratio
        zxy1 = project_data.get_centroids_as_numpy(i_volume)
        if len(zxy1) == 0:
            continue
        # zxy1[:, 0] *= z_to_xy_ratio
        # Get matches
        matches, conf, = calc_nearest_neighbor_matches(zxy0, zxy1, max_dist=max_dist)

        def seg_array_to_mask_ind(i):
            # The seg_zxy array has the 0th row corresponding to segmentation mask label 1
            # BUT can also skip rows and might generally be non-monotonic
            return project_data.segmentation_metadata.seg_array_to_mask_index(i_volume, i)

        def dlc_array_to_ind(i):
            # the 0th index corresponds to neuron_001, and should finally be label 1
            return i + 1

        # Save
        all_matches[i_volume] = np.array(
            [[dlc_array_to_ind(m[0]), seg_array_to_mask_ind(m[1]), c] for m, c in zip(matches, conf)]
        )


def _unpack_configs_for_traces(project_cfg, segment_cfg, track_cfg):
    # Settings
    max_dist = track_cfg.config['final_3d_tracks']['max_dist_to_segmentation']
    params_start_volume = project_cfg.config['dataset_params']['start_volume']
    num_frames = project_cfg.config['dataset_params']['num_frames']
    dlc_fname = track_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
    z_to_xy_ratio = project_cfg.config['dataset_params']['z_to_xy_ratio']
    green_fname = project_cfg.config['preprocessed_green']
    red_fname = project_cfg.config['preprocessed_red']

    dlc_tracks: pd.DataFrame = pd.read_hdf(dlc_fname)

    return dlc_tracks, green_fname, red_fname, max_dist, num_frames, params_start_volume, z_to_xy_ratio


def _unpack_configs_for_extraction(project_cfg, traces_cfg):
    # Settings
    params_start_volume = project_cfg.config['dataset_params']['start_volume']
    num_frames = project_cfg.config['dataset_params']['num_frames']

    frame_list = list(range(params_start_volume, num_frames + params_start_volume))

    coords = ['z', 'x', 'y']
    fname = traces_cfg.resolve_relative_path_from_config('reindexed_masks')
    reindexed_masks = zarr.open(fname)

    return coords, reindexed_masks, frame_list, params_start_volume


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
