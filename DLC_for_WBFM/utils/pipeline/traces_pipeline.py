import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
import zarr
from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_from_distance
from DLC_for_WBFM.utils.feature_detection.visualization_tracks import visualize_tracks
from DLC_for_WBFM.utils.projects.utils_project import edit_config
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume


def get_traces_from_3d_tracks(segment_cfg,
                              track_cfg,
                              traces_cfg,
                              project_cfg,
                              DEBUG=False):
    """
    Connect the 3d traces to previously segmented masks

    Get both red and green traces for each neuron
    """
    # Settings
    max_dist = track_cfg['final_3d_tracks']['max_dist_to_segmentation']
    start_volume = project_cfg['dataset_params']['start_volume']
    num_frames = project_cfg['dataset_params']['num_frames']
    # Get previous annotations
    segmentation_fname = segment_cfg['output']['metadata']
    with open(segmentation_fname, 'rb') as f:
        segmentation_metadata = pickle.load(f)
    dlc_fname = track_cfg['final_3d_tracks']['df_fname']
    dlc_tracks = pd.read_hdf(dlc_fname)

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
    frame_list = list(range(start_volume, num_frames + start_volume))
    green_dat, red_dat = _initialize_dataframes(all_neuron_names, frame_list)

    all_matches = {}  # key = i_vol; val = Nx3-element list
    print("Matching segmentation and DLC tracking...")
    for i_volume in tqdm(frame_list):
        # Get DLC point cloud
        # NOTE: This dataframe starts at 0, not start_volume
        zxy0 = _get_dlc_zxy(i_volume)
        zxy0[:, 0] *= project_cfg['dataset_params']['z_to_xy_ratio']
        # REVIEW: Get segmentation point cloud
        seg_zxy = segmentation_metadata[i_volume]['centroids']
        seg_zxy = [np.asarray(row) for row in seg_zxy]
        zxy1 = np.array(seg_zxy)
        zxy1[:, 0] *= project_cfg['dataset_params']['z_to_xy_ratio']
        # Get matches
        out = calc_bipartite_from_distance(zxy0, zxy1, max_dist=max_dist)
        matches, conf, _ = out
        if DEBUG:
            visualize_tracks(zxy0, zxy1, matches)
        # Use metadata to get red traces
        # OPTIMIZE: minimum confidence?
        mdat = segmentation_metadata[i_volume]
        all_seg_names = list(mdat['centroids'].keys())
        zxy0[:, 0] /= project_cfg['dataset_params']['z_to_xy_ratio']  # Return to original
        for i_dlc, i_seg in matches:
            d_name = all_neuron_names[i_dlc]  # output name
            s_name = int(all_seg_names[i_seg])
            # See saved_names above
            i = i_volume
            red_dat[(d_name, 'brightness')].loc[i] = mdat['total_brightness'][s_name]
            red_dat[(d_name, 'volume')].loc[i] = mdat['neuron_volume'][s_name]
            red_dat[(d_name, 'centroid_ind')].loc[i] = s_name
            zxy_seg = mdat['centroids'][s_name]
            zxy_dlc = zxy0[i_dlc]
            _save_locations_in_df(d_name, red_dat, i, zxy_dlc, zxy_seg)
            if DEBUG:
                break

        # Save
        all_matches[i_volume] = np.hstack([matches, conf])
        if DEBUG:
            break

    print("Extracting green traces...")
    green_fname = project_cfg['green_bigtiff_fname']
    num_slices = project_cfg['dataset_params']['num_slices']
    mask_array = zarr.open(segment_cfg['output']['masks'])
    vol_opt = {'num_slices': num_slices, 'dtype': 'uint16'}
    with tifffile.TiffFile(green_fname) as green_tifffile:
        for i_volume in tqdm(frame_list):
            # Prepare matches and locations
            matches = all_matches[i_volume]
            mdat = segmentation_metadata[i_volume]
            all_seg_names = list(mdat['centroids'].keys())
            all_zxy_dlc = _get_dlc_zxy(i_volume)
            # Prepare mask (segmentation)
            i_mask = i_volume - project_cfg['dataset_params']['start_volume']
            # this_mask_volume = get_single_volume(mask_fname, i_mask, **vol_opt) # TODO: can this read zarr directly?
            this_mask_volume = mask_array[i_mask, ...]
            this_green_volume = get_single_volume(green_tifffile, i_volume, **vol_opt)
            is_mirrored = project_cfg['dataset_params']['red_and_green_mirrored']
            for i_dlc, i_seg, _ in matches:
                _analyze_video_using_mask(all_neuron_names, all_seg_names, all_zxy_dlc, green_dat, i_dlc, i_seg, i_volume,
                                          is_mirrored, mdat, this_green_volume, this_mask_volume)
                if DEBUG:
                    print("Single pass-through sucessful; did not write any files")
                    return

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


def _initialize_dataframes(all_neuron_names, frame_list):
    save_names = ['brightness', 'volume',
                  'centroid_ind',
                  'z_seg', 'x_seg', 'y_seg',
                  'z_dlc', 'x_dlc', 'y_dlc']
    m_index = pd.MultiIndex.from_product([all_neuron_names,
                                          save_names],
                                         names=['neurons', 'data'])
    sz = (len(frame_list), len(m_index))
    empty_dat = np.empty(sz)
    empty_dat[:] = np.nan
    red_dat = pd.DataFrame(empty_dat,
                           columns=m_index,
                           index=frame_list)
    green_dat = red_dat.copy()
    return green_dat, red_dat


def _analyze_video_using_mask(all_neuron_names, all_seg_names, all_zxy_dlc, green_dat, i_dlc, i_seg, i_volume,
                              is_mirrored, mdat, this_green_volume, this_mask_volume):
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
    brightness = np.sum(this_green_volume[this_mask_neuron])
    # Save in dataframe
    green_dat[(d_name, 'brightness')].loc[i] = brightness
    green_dat[(d_name, 'volume')].loc[i] = volume
    green_dat[(d_name, 'centroid_ind')].loc[i] = s_name
    zxy_seg = mdat['centroids'][s_name]
    zxy_dlc = all_zxy_dlc[i_dlc]
    _save_locations_in_df(d_name, green_dat, i, zxy_dlc, zxy_seg)


def _save_locations_in_df(d_name, df, i, zxy_dlc, zxy_seg):
    df[(d_name, 'z_seg')].loc[i] = zxy_seg[0]
    df[(d_name, 'x_seg')].loc[i] = zxy_seg[1]
    df[(d_name, 'y_seg')].loc[i] = zxy_seg[2]
    df[(d_name, 'z_dlc')].loc[i] = zxy_dlc[0]
    df[(d_name, 'x_dlc')].loc[i] = zxy_dlc[1]
    df[(d_name, 'y_dlc')].loc[i] = zxy_dlc[2]