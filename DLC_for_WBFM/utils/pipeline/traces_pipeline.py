import pickle
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Callable, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_from_distance
from DLC_for_WBFM.utils.projects.utils_project import edit_config


def get_traces_from_3d_tracks_using_config(segment_cfg: dict,
                                           track_cfg: dict,
                                           traces_cfg: dict,
                                           project_cfg: dict,
                                           DEBUG: bool = False) -> None:
    """
    Connect the 3d traces to previously segmented masks

    Get both red and green traces for each neuron
    """
    dlc_tracks, green_fname, is_mirrored, mask_array, max_dist, num_frames, params_start_volume, segmentation_metadata, z_to_xy_ratio = _unpack_configs_for_traces(
        project_cfg, segment_cfg, track_cfg)

    # DEPRECATE preprocessing
    green_video = zarr.open(green_fname)
    # cfg = project_cfg.copy()
    # cfg['preprocessing_config'] = track_cfg['preprocessing_config']
    # green_video, _ = preprocess_all_frames_using_config(DEBUG, cfg, verbose=0, vid_fname=green_fname)

    all_matches, all_neuron_names, green_dat, red_dat = get_traces_from_3d_tracks(DEBUG, dlc_tracks, green_video,
                                                                                  is_mirrored, mask_array, max_dist,
                                                                                  num_frames,
                                                                                  params_start_volume,
                                                                                  segmentation_metadata, z_to_xy_ratio)

    if DEBUG:
        print("Single pass-through sucessful; did not write any files")
        return

    _save_traces_as_hdf_and_update_configs(all_matches, all_neuron_names, green_dat, red_dat, traces_cfg)


def _save_traces_as_hdf_and_update_configs(all_matches: defaultdict, all_neuron_names: list,
                                           green_dat: pd.DataFrame, red_dat: pd.DataFrame, traces_cfg: dict) -> None:
    # Save traces (red and green) and neuron names
    red_fname = Path('4-traces').joinpath('red_traces.h5')
    red_dat.to_hdf(red_fname, "df_with_missing")
    green_fname = Path('4-traces').joinpath('green_traces.h5')
    green_dat.to_hdf(green_fname, "df_with_missing")
    # Also save matches as a separate file
    # ENHANCE: save as part of the dataframes?
    matches_fname = Path('4-traces').joinpath('all_matches.pickle')
    with open(matches_fname, 'wb') as f:
        pickle.dump(all_matches, f)
    # Save the output filenames
    traces_cfg['all_matches'] = str(matches_fname)
    traces_cfg['traces']['green'] = str(green_fname)
    traces_cfg['traces']['red'] = str(red_fname)
    traces_cfg['traces']['neuron_names'] = all_neuron_names
    edit_config(traces_cfg['self_path'], traces_cfg)


def get_traces_from_3d_tracks(DEBUG: bool, dlc_tracks: pd.DataFrame, green_video: zarr.Array, is_mirrored: bool,
                              mask_array: zarr.Array, max_dist: float, num_frames: int,
                              params_start_volume: int, segmentation_metadata: dict,
                              z_to_xy_ratio: float) -> Tuple[defaultdict, list, pd.DataFrame, pd.DataFrame]:
    # Convert DLC dataframe to array
    all_neuron_names = list(dlc_tracks.columns.levels[0])

    def _get_dlc_zxy(t, dlc_tracks=dlc_tracks):
        all_dlc_zxy = np.zeros((len(all_neuron_names), 3))
        coords = ['z', 'y', 'x']
        for i, name in enumerate(all_neuron_names):
            all_dlc_zxy[i, :] = np.asarray(dlc_tracks[name][coords].loc[t])
        return all_dlc_zxy

    # Main loop: Match segmentations to tracks
    # Also: get connected red brightness and mask
    # Initialize multi-index dataframe for data
    frame_list = list(range(params_start_volume, num_frames + params_start_volume))
    green_dat, red_dat = _initialize_dataframes(all_neuron_names, frame_list)
    all_matches = defaultdict(list)  # key = i_vol; val = Nx3-element list
    print("Matching segmentation and DLC tracking...")
    if DEBUG:
        frame_list = frame_list[:2]  # Shorten (to avoid break)
    calculate_segmentation_and_dlc_matches(_get_dlc_zxy, all_matches, all_neuron_names, frame_list, max_dist, red_dat,
                                           segmentation_metadata, z_to_xy_ratio)

    print("Extracting green traces using matches...")
    for i_volume in tqdm(frame_list):
        # Prepare matches and locations
        matches = all_matches[i_volume]
        if len(matches) == 0:
            continue
        mdat = segmentation_metadata[i_volume]
        all_seg_names = list(mdat['centroids'].keys())
        all_zxy_dlc = _get_dlc_zxy(i_volume)
        # Prepare mask (segmentation)
        i_mask = i_volume - params_start_volume
        this_mask_volume = mask_array[i_mask, ...]
        this_green_volume = green_video[i_volume, ...]
        for i_dlc, i_seg, c in matches:
            i_dlc, i_seg = i_dlc - 1, i_seg - 1  # Matches start at 1
            _analyze_video_using_mask(all_neuron_names, all_seg_names, all_zxy_dlc, green_dat, i_dlc, i_seg,
                                      i_volume,
                                      is_mirrored, mdat, this_green_volume, this_mask_volume, c)

    return all_matches, all_neuron_names, green_dat, red_dat


def calculate_segmentation_and_dlc_matches(_get_dlc_zxy: Callable,
                                           all_matches: defaultdict,
                                           all_neuron_names: list,
                                           frame_list: list,
                                           max_dist: float,
                                           red_dat: pd.DataFrame,
                                           segmentation_metadata: Dict[int, pd.DataFrame],
                                           z_to_xy_ratio: float) -> None:
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
        out = calc_bipartite_from_distance(zxy0, zxy1, max_dist=max_dist)
        matches, conf, _ = out
        # if DEBUG:
        #     visualize_tracks(zxy0, zxy1, matches)
        # Use metadata to get red traces
        # OPTIMIZE: minimum confidence?
        mdat = segmentation_metadata[i_volume]
        all_seg_names = list(mdat['centroids'].keys())
        zxy0[:, 0] /= z_to_xy_ratio  # Return to original
        for (i_dlc, i_seg), c in zip(matches, conf):
            d_name = all_neuron_names[i_dlc]  # output name
            s_name = int(all_seg_names[i_seg])
            # See saved_names above
            i = i_volume
            if 'all_values' in mdat:
                red_dat[(d_name, 'all_values')].loc[i] = mdat['all_values'][s_name]
            elif 'pixel_counts' in mdat:
                # Temporary workaround when I saved the wrong thing
                red_dat[(d_name, 'all_values')].loc[i] = [rebuild_pixel_values(mdat, s_name)]
            red_dat[(d_name, 'brightness')].loc[i] = mdat['total_brightness'][s_name]
            red_dat[(d_name, 'volume')].loc[i] = mdat['neuron_volume'][s_name]
            red_dat[(d_name, 'centroid_ind')].loc[i] = s_name
            zxy_seg = mdat['centroids'][s_name]
            zxy_dlc = zxy0[i_dlc]
            _save_locations_in_df(d_name, red_dat, i, zxy_dlc, zxy_seg, c)

        # Save
        # all_matches[i_volume] = np.hstack([matches, conf])
        # NOTE: need to offset by 1, because the background is 0
        all_matches[i_volume] = np.array([[m[0] + 1, m[1] + 1, c] for m, c in zip(matches, conf)])


def _unpack_configs_for_traces(project_cfg, segment_cfg, track_cfg):
    # Settings
    max_dist = track_cfg['final_3d_tracks']['max_dist_to_segmentation']
    params_start_volume = project_cfg['dataset_params']['start_volume']
    num_frames = project_cfg['dataset_params']['num_frames']
    # Get previous annotations
    segmentation_fname = segment_cfg['output']['metadata']
    with open(segmentation_fname, 'rb') as f:
        segmentation_metadata = pickle.load(f)
    dlc_fname = track_cfg['final_3d_tracks']['df_fname']
    z_to_xy_ratio = project_cfg['dataset_params']['z_to_xy_ratio']
    # green_fname = project_cfg['green_bigtiff_fname']
    green_fname = project_cfg['preprocessed_green']
    # num_slices = project_cfg['dataset_params']['num_slices']
    mask_array = zarr.open(segment_cfg['output']['masks'])
    is_mirrored = project_cfg['dataset_params']['red_and_green_mirrored']

    dlc_tracks: pd.DataFrame = pd.read_hdf(dlc_fname)

    return dlc_tracks, green_fname, is_mirrored, mask_array, max_dist, num_frames, params_start_volume, segmentation_metadata, z_to_xy_ratio


def _initialize_dataframes(all_neuron_names: List[str], frame_list: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    save_names = ['brightness', 'volume',
                  'all_values',
                  'centroid_ind',
                  'z_seg', 'x_seg', 'y_seg',
                  'z_dlc', 'x_dlc', 'y_dlc',
                  'match_confidence']
    m_index = pd.MultiIndex.from_product([all_neuron_names,
                                          save_names],
                                         names=['neurons', 'data'])
    sz = (len(frame_list), len(m_index))
    empty_dat = np.empty(sz)
    empty_dat[:] = np.nan
    red_dat = pd.DataFrame(empty_dat,
                           columns=m_index,
                           index=frame_list)
    for name in all_neuron_names:
        # Allow saving numpy arrays in the column
        red_dat[(name, 'all_values')] = red_dat[(name, 'all_values')].astype('object')
    green_dat = red_dat.copy()
    return green_dat, red_dat


def _analyze_video_using_mask(all_neuron_names: List[str], all_seg_names: list, all_zxy_dlc: np.ndarray,
                              green_dat: pd.DataFrame, i_dlc: int, i_seg: int, i_volume: int,
                              is_mirrored: bool, mdat: dict, this_green_volume: np.ndarray,
                              this_mask_volume: np.ndarray, confidence: float) -> None:
    # For conversion between lists
    i_dlc, i_seg = int(i_dlc), int(i_seg)
    d_name = all_neuron_names[i_dlc]  # output name
    s_name = int(all_seg_names[i_seg])
    i = i_volume
    # Get brightness from green volume and mask
    this_mask_neuron = (this_mask_volume == s_name)
    if is_mirrored:
        this_mask_neuron = np.flip(this_mask_neuron, axis=2)
    volume = np.count_nonzero(this_mask_neuron)
    all_values = this_green_volume[this_mask_neuron]
    brightness = np.sum(all_values)
    # Save in dataframe
    green_dat[(d_name, 'brightness')].loc[i] = brightness
    green_dat[(d_name, 'volume')].loc[i] = volume
    green_dat[(d_name, 'centroid_ind')].loc[i] = s_name
    green_dat[(d_name, 'all_values')].loc[i] = all_values
    zxy_seg = mdat['centroids'][s_name]
    zxy_dlc = all_zxy_dlc[i_dlc]
    _save_locations_in_df(d_name, green_dat, i, zxy_dlc, zxy_seg, confidence)


def _save_locations_in_df(d_name, df, i, zxy_dlc, zxy_seg, conf):
    df[(d_name, 'z_seg')].loc[i] = zxy_seg[0]
    df[(d_name, 'x_seg')].loc[i] = zxy_seg[1]
    df[(d_name, 'y_seg')].loc[i] = zxy_seg[2]
    df[(d_name, 'z_dlc')].loc[i] = zxy_dlc[0]
    df[(d_name, 'x_dlc')].loc[i] = zxy_dlc[1]
    df[(d_name, 'y_dlc')].loc[i] = zxy_dlc[2]
    df[(d_name, 'match_confidence')].loc[i] = conf


def rebuild_pixel_values(frame_df, which_neuron):
    # I accidentally didn't save the full histogram... so I remake it
    counts, edges = frame_df.loc[which_neuron, 'pixel_counts'], frame_df.loc[which_neuron, 'pixel_values']
    pixel_vals = np.repeat(edges, counts)
    return pixel_vals