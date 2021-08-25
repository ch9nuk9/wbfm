import napari
import numpy as np
from PyQt5 import QtGui

from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords3d
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


def zoom_using_viewer(viewer: napari.Viewer, layer_name='pts_with_future_and_past', zoom=None) -> None:
    # Get current point
    t = viewer.dims.current_step[0]
    tzxy = viewer.layers[layer_name].data[t]

    # Data may be incorrect (but t should be good)
    is_positive = tzxy[2] > 0 and tzxy[3] > 0
    is_finite = not all(np.isnan(tzxy))

    # Center to the neuron in xy
    if zoom is not None:
        viewer.camera.zoom = zoom
    if is_positive and is_finite:
        viewer.camera.center = tzxy[1:]

    # Center around the neuron in z
    if is_positive and is_finite:
        viewer.dims.current_step = (t, tzxy[1], 0, 0)


def change_viewer_time_point(viewer: napari.Viewer, dt: int, a_max: int = None) -> None:
    # Increment time
    t = np.clip(viewer.dims.current_step[0] + dt, a_min=0, a_max=a_max)
    tzxy = (t,) + viewer.dims.current_step[1:]
    viewer.dims.current_step = tzxy
