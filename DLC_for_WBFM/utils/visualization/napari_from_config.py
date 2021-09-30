import os

import napari
import numpy as np
import zarr
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config
from DLC_for_WBFM.utils.projects.utils_project import safe_cd


def napari_of_training_data(cfg: modular_project_config):
    # TODO: read from config file, not project directory
    training_dat_fname = cfg.config['preprocessed_red']
    training_seg_fname = os.path.join('2-training_data', 'reindexed_masks.zarr')
    with safe_cd(cfg.project_dir):
        z_dat = zarr.open_array(training_dat_fname)
        z_seg = zarr.open_array(training_seg_fname)

    viewer = napari.view_labels(z_seg, ndisplay=3)
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
    if neuron_name_dict is None:
        neuron_name_dict = {}
    all_neurons = list(df.columns.levels[0])
    i_name = 'i_reindexed_segmentation'
    zxy_names = ['z_dlc', 'x_dlc', 'y_dlc']
    t_vec = np.expand_dims(np.array(list(df.index), dtype=int), axis=1)
    # label_vec = np.ones(len(df.index), dtype=int)
    all_t_zxy = np.array([[0, 0, 0, 0]], dtype=int)
    properties = {'label': []}
    for n in all_neurons:
        zxy = df[n][zxy_names].to_numpy(dtype=int)
        t_zxy = np.hstack([t_vec, zxy])
        if n in neuron_name_dict:
            # label_vec[:] = this_name
            label_vec = [neuron_name_dict[n]] * len(df.index)
            if DEBUG:
                print(f"Found named neuron: {n} = {label_vec[0]}")
        else:
            label_vec = list(df[n][i_name])

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
