import logging

import napari
import numpy as np
from PyQt5 import QtGui

from DLC_for_WBFM.utils.general.postprocessing.base_cropping_utils import get_crop_coords3d
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume, \
    get_single_volume_specific_slices


def get_cropped_frame(fname, t, num_slices, zxy, crop_sz, to_flip=False):
    # print(f"Loading file: {fname}")
    # print(f"Location: {zxy}")
    if crop_sz is not None:
        crop_coords = get_crop_coords3d(zxy, crop_sz)
        z, x, y = crop_coords
        if len(z) > 1:
            start_slice, end_slice = z[0], z[-1]
        else:
            start_slice, end_slice = z[0], z[0] + 1
        dat = get_single_volume_specific_slices(fname, t, num_slices,
                                                start_slice, end_slice)
        if to_flip:
            dat = np.flip(dat, axis=1)
        # print(f"crop_coords: {crop_coords}")
        dat_crop = dat[x[0]:x[-1], y[0]:y[-1]]
    else:
        dat_crop = get_single_volume(fname, t, num_slices, dtype='uint16')
        if to_flip:
            dat_crop = np.flip(dat_crop, axis=2)

    return _fix_dimension_for_plt(crop_sz, dat_crop)


def _fix_dimension_for_plt(crop_sz, dat_crop):
    # Final output should be XYC
    if len(dat_crop.shape) == 3:
        if crop_sz is None:
            # Just visualize center of worm
            dat_crop = dat_crop[15]  # Remove z
        else:
            dat_crop = dat_crop[0]
    return np.array(dat_crop)


def get_crop_from_zarr(zarr_array, t, zxy, crop_sz):
    if crop_sz is not None:
        crop_coords = get_crop_coords3d(zxy, crop_sz)
        z, x, y = crop_coords
        if len(z) > 1:
            start_slice, end_slice = z[0], z[-1]
        else:
            start_slice, end_slice = z[0], z[0] + 1
        # print(f"Zarr size before crop: {zarr_array.shape}")
        this_volume = np.array(zarr_array[t, ...])
        dat_crop = this_volume[start_slice:end_slice, x[0]:x[-1], y[0]:y[-1]]
        # print(f"Zarr size after crop: {dat_crop.shape}")
    else:
        dat_crop = zarr_array[t, :, :, :]

    return _fix_dimension_for_plt(crop_sz, dat_crop)


def array2qt(img):
    # From: https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
    h, w, channel = img.shape
    # bytesPerLine = 3 * w
    # return QtGui.QPixmap(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    new_img = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(new_img)


def zoom_using_viewer(viewer: napari.Viewer, layer_name='pts_with_future_and_past', zoom=None,
                      layer_is_full_size_and_single_neuron=True, ind_within_layer=None) -> None:
    # Get current point
    t = viewer.dims.current_step[0]
    if layer_is_full_size_and_single_neuron:
        tzxy = get_zxy_from_single_neuron_layer(viewer.layers[layer_name], t)
    else:
        tzxy = get_zxy_from_multi_neuron_layer(viewer.layers[layer_name], t, ind_within_layer)
    print(f"tzxy: {tzxy}, layer: {viewer.layers[layer_name]}")
    # TODO: better way to check for nesting
    if len(tzxy) == 1:
        tzxy = tzxy[0]
    if len(tzxy) == 1:
        tzxy = tzxy[0]
    # Data may be actually a null value (but t should be good)
    try:
        is_positive = tzxy[2] > 0 and tzxy[3] > 0
        is_finite = not all(np.isnan(tzxy))
    except IndexError:
        logging.warning("Index error in zooming; skipping")
        return

    # Center to the neuron in xy
    if zoom is not None:
        viewer.camera.zoom = zoom
    if is_positive and is_finite:
        viewer.camera.center = tzxy[1:]

    # Center around the neuron in z
    if is_positive and is_finite:
        viewer.dims.current_step = (t, tzxy[1], 0, 0)


def get_zxy_from_single_neuron_layer(layer, t, ind_within_layer=None):
    return layer.data[t]


def get_zxy_from_multi_neuron_layer(layer, t, ind_within_layer=None):
    # e.g. text labels, with all neurons in a time point in a row (thus t is no longer a direct index)
    # Or, if nans have been dropped from an otherwise full-size layer
    # Note: if ind_within_layer is None, it has no effect
    dat = layer.data
    if dat.shape[1] == 5:
        # Tracks layer; neuron index is now first column
        dat = dat[:, 1:]
    elif dat.shape[1] == 4:
        # Points layer
        pass
    else:
        raise ValueError(f"Unrecognized layer shape {dat.shape}")
    ind = dat[:, 0] == t

    if len(np.where(ind)[0]) == 0:
        fake_dat = np.zeros_like(layer.data[0, :])
        fake_dat[0] = t
        # logging.warning(f"Time {t} not found in layer: {layer.data[:, 0]}")
        return fake_dat
    else:
        if ind_within_layer is not None:
            return layer.data[ind, :][ind_within_layer, :]
        else:
            return layer.data[ind, :]


def change_viewer_time_point(viewer: napari.Viewer,
                             dt: int = None, t_target: int=None, a_max: int = None) -> None:
    # Increment time
    if dt is not None:
        t = np.clip(viewer.dims.current_step[0] + dt, a_min=0, a_max=a_max)
    elif t_target is not None:
        t = np.clip(t_target, a_min=0, a_max=a_max)
    else:
        raise ValueError("Must pass either target time or dt")
    tzxy = (t,) + viewer.dims.current_step[1:]
    viewer.dims.current_step = tzxy


def build_tracks_from_dataframe(df_single_track, likelihood_thresh=None):
    # Just visualize one neuron for now
    # 5 columns:
    # track_id, t, z, y, x
    try:
        coords = ['z', 'x', 'y']
        zxy_array = np.array(df_single_track[coords])
    except KeyError:
        coords = ['z_dlc', 'x_dlc', 'y_dlc']
        zxy_array = np.array(df_single_track[coords])

    all_tracks_list = []
    t_array = np.expand_dims(np.arange(zxy_array.shape[0]), axis=1)

    if likelihood_thresh is not None and 'likelihood' in df_single_track:
        to_remove = df_single_track['likelihood'] < likelihood_thresh
    else:
        to_remove = np.zeros_like(zxy_array[:, 0], dtype=bool)
    zxy_array[to_remove, :] = 0

    # Also remove values that are entirely nan
    rows_not_nan = ~(np.isnan(zxy_array)[:, 0])
    zxy_array = zxy_array[rows_not_nan, :]
    t_array = t_array[rows_not_nan, :]

    all_tracks_list.append(np.hstack([t_array, zxy_array]))
    all_tracks_array = np.vstack(all_tracks_list)

    track_of_point = np.hstack([np.ones((all_tracks_array.shape[0], 1)), all_tracks_array])

    return all_tracks_array, track_of_point, to_remove
