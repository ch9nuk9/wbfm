import logging
import os
from typing import Tuple

import napari
import numpy as np
import zarr

from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd


def napari_of_training_data(cfg: ModularProjectConfig) -> Tuple[napari.Viewer, np.ndarray, np.ndarray]:

    project_data = ProjectData.load_final_project_data_from_config(cfg)
    # training_dat_fname = cfg.config['preprocessed_red']
    training_cfg = cfg.get_training_config()
    # training_seg_fname = training_cfg.resolve_relative_path_from_config('reindexed_masks')
    #
    # segmentation_cfg = cfg.get_segmentation_config()
    # raw_seg_fname = segmentation_cfg.resolve_relative_path_from_config('output_masks')
    #
    # z_dat = zarr.open_array(training_dat_fname)
    # z_seg = zarr.open_array(training_seg_fname)
    # raw_seg = zarr.open(raw_seg_fname)

    z_dat = project_data.red_data
    raw_seg = project_data.segmentation
    z_seg = project_data.reindexed_masks_training

    # Training data doesn't usually start at i=0, so align
    num_frames = training_cfg.config['training_data_3d']['num_training_frames']
    i_seg_start = training_cfg.config['training_data_3d']['which_frames'][0]
    i_seg_end = i_seg_start + num_frames
    z_dat = z_dat[i_seg_start:i_seg_end, ...]
    raw_seg = raw_seg[i_seg_start:i_seg_end, ...]

    logging.info(f"Size of reindexed_masks: {z_dat.shape}")

    viewer = napari.view_labels(z_seg, ndisplay=3)
    viewer.add_labels(raw_seg, visible=False)
    viewer.add_image(z_dat)
    viewer.show()

    return viewer, z_dat, z_seg


def napari_of_full_data(project_dir):
    # TODO: read from config file, not project directory
    training_dat_fname = os.path.join('4-traces', 'data_red_channel.zarr')
    dat_exists = os.path.exists(training_dat_fname)
    training_seg_fname = os.path.join('4-traces', 'reindexed_masks.zarr')
    with safe_cd(project_dir):
        if dat_exists:
            z_dat = zarr.open_array(training_dat_fname)
        else:
            z_dat = None

        if not os.path.exists(training_seg_fname):
            raise FileNotFoundError(f"{training_seg_fname} must exist; run scripts/visualization/4+reindex_...")
        z_seg = zarr.open_array(training_seg_fname)

    viewer = napari.view_labels(z_seg, ndisplay=3)
    if dat_exists:
        viewer.view_data(z_dat)
    viewer.show()

    return viewer, z_dat, z_seg


def dlc_to_napari_tracks(df, likelihood_thresh=0.4):
    """
    Convert a deeplabcut-style track to an array that can be visualized using:
        napari.view_tracks(dat)
    """

    # Convert tracks to napari style
    neuron_names = df.columns.remove_unused_levels().levels[0]
    # 5 columns:
    # track_id, t, z, y, x
    coords = ['z', 'y', 'x']
    all_tracks_list = []
    for i, name in enumerate(neuron_names):
        zxy_array = np.array(df[name][coords])
        t_array = np.expand_dims(np.arange(zxy_array.shape[0]), axis=1)

        # Remove low likelihood
        if 'likelihood' in df[name]:
            to_keep = df[name]['likelihood'] > likelihood_thresh
            zxy_array = zxy_array[to_keep, :]
            t_array = t_array[to_keep, :]
        id_array = np.ones_like(t_array) * i

        all_tracks_list.append(np.hstack([id_array, t_array, zxy_array]))

    return np.vstack(all_tracks_list)


def napari_tracks_from_match_list(list_of_matches, n0_zxy_raw, n1_zxy_raw):
    all_tracks_list = []
    for i_track, m in enumerate(list_of_matches):
        track_m0 = [i_track, 0]
        track_m0.extend(n0_zxy_raw[m[0]])

        track_m1 = [i_track, 1]
        track_m1.extend(n1_zxy_raw[m[1]])

        all_tracks_list.append(track_m0)
        all_tracks_list.append(track_m1)
    return all_tracks_list


def napari_labels_from_traces_dataframe(df, neuron_name_dict=None, DEBUG=False):
    """
    Expected napari-ready format:
        A dict of options, with a nested dict 'properties' and a list 'data'
        'properties' has one entry, 'labels' = long list with all points at all time
        'dat' is a list of equal length with all the dimensions (tzxy)

    Parameters
    ----------
    df
    neuron_name_dict
    DEBUG

    Returns
    -------

    """
    df.replace(0, np.NaN, inplace=True)  # DLC uses all zeros as failed tracks

    if neuron_name_dict is None:
        neuron_name_dict = {}
    all_neurons = list(df.columns.levels[0])
    t_vec = np.expand_dims(np.array(list(df.index), dtype=int), axis=1)
    # label_vec = np.ones(len(df.index), dtype=int)
    all_t_zxy = np.array([[0, 0, 0, 0]], dtype=int)
    properties = {'label': []}
    for n in all_neurons:
        try:
            coords = ['z_dlc', 'x_dlc', 'y_dlc']
            zxy = np.array(df[n][coords])
        except KeyError:
            coords = ['z', 'x', 'y']
            zxy = np.array(df[n][coords])
        # zxy = df[n][zxy_names].to_numpy(dtype=int)
        t_zxy = np.hstack([t_vec, zxy])
        if n in neuron_name_dict:
            # label_vec[:] = this_name
            label_vec = [neuron_name_dict[n]] * len(df.index)
            if DEBUG:
                print(f"Found named neuron: {n} = {label_vec[0]}")
        else:
            try:
                i_name = 'i_reindexed_segmentation'
                label_vec = list(df[n][i_name])
            except KeyError:
                i_name = 'label'
                label_vec = [cast_int_or_nan(i) for i in df[n][i_name]]

        all_t_zxy = np.vstack([all_t_zxy, t_zxy])
        properties['label'].extend(label_vec)
    all_t_zxy = np.where(all_t_zxy < 0, np.nan, all_t_zxy)  # Some points are negative instead of nan
    to_keep = ~np.isnan(all_t_zxy).any(axis=1)
    all_t_zxy = all_t_zxy[to_keep, :]
    all_t_zxy = all_t_zxy[1:, :]  # Remove dummy starter point
    properties['label'] = [p for p, good in zip(properties['label'], to_keep[1:]) if good]

    options = {'data': all_t_zxy, 'face_color': 'transparent', 'edge_color': 'transparent', 'text': 'label',
               'properties': properties, 'name': 'Neuron IDs'}

    return options


def cast_int_or_nan(i):
    if np.isnan(i):
        return i
    else:
        return int(i)


def napari_labels_from_frames(all_frames: dict, num_frames=1) -> dict:

    all_t_zxy = np.array([[0, 0, 0, 0]], dtype=int)
    properties = {'label': []}
    for i_frame, frame in all_frames.items():
        if i_frame >= num_frames:
            break
        zxy = frame.neuron_locs
        zxy = zxy[:, [0, 2, 1]]
        num_neurons = zxy.shape[0]
        t_vec = np.ones((num_neurons, 1)) * i_frame
        t_zxy = np.hstack([t_vec, zxy])

        label_vec = list(range(num_neurons))

        all_t_zxy = np.vstack([all_t_zxy, t_zxy])
        properties['label'].extend(label_vec)

    all_t_zxy = all_t_zxy[1:, :]  # Remove dummy starter point
    options = {'data': all_t_zxy, 'face_color': 'transparent', 'edge_color': 'transparent', 'text': 'label',
               'properties': properties, 'name': 'Raw IDs'}

    return options
