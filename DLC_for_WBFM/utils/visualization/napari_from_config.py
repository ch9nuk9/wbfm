import os

import numpy as np
import zarr
import napari
from DLC_for_WBFM.utils.projects.utils_project import safe_cd


def napari_of_training_data(project_dir):
    # TODO: read from config file, not project directory
    training_dat_fname = os.path.join('2-training_data', 'training_data_red_channel.zarr')
    training_seg_fname = os.path.join('2-training_data', 'reindexed_masks.zarr')
    with safe_cd(project_dir):
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


def create_text_labels_for_napari(df):
    all_neurons = list(df.columns.levels[0])
    i_name = 'i_reindexed_segmentation'
    zxy_names = ['z_dlc', 'x_dlc', 'y_dlc']
    t_vec = np.expand_dims(np.array(list(df.index), dtype=int), axis=1)
    label_vec = np.ones(len(df.index), dtype=int)
    pts = np.array([[0, 0, 0, 0]], dtype=int)
    properties = {'label': []}
    for n in all_neurons:
        zxy = df[n][zxy_names].to_numpy(dtype=int)
        t_zxy = np.hstack([t_vec, zxy])
        label_vec[:] = df[n][i_name]

        pts = np.vstack([pts, t_zxy])
        properties['label'].extend(label_vec)
    pts = np.where(pts < 0, np.nan, pts)  # Some points are negative instead of nan
    to_keep = ~np.isnan(pts).any(axis=1)
    pts = pts[to_keep, :]
    pts = pts[1:, :]
    properties['label'] = [p for p, good in zip(properties['label'], to_keep[1:]) if good]

    options = {'data': pts, 'face_color': 'transparent', 'edge_color': 'transparent', 'text': 'label',
           'properties': properties, 'name': 'Neuron IDs'}

    return options