from PyQt5 import QtGui
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords3d
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
import numpy as np


def get_cropped_frame(fname, t, num_slices, zxy, crop_sz):
    # print(f"Loading file: {fname}")
    # print(f"Location: {zxy}")
    crop_coords = get_crop_coords3d(zxy, crop_sz)
    z, x, y = crop_coords
    # print(f"crop_coords: {crop_coords}")
    dat = get_single_volume(fname, t, num_slices, dtype='uint16')
    dat_crop = dat[z, x[0]:x[-1], y[0]:y[-1]]
    # Final output should be XYC
    if len(dat_crop.shape) == 3:
        dat_crop = dat_crop[0]  # Remove z
    # dat_crop = np.expand_dims(dat_crop, axis=-1)  # Add color channel
    return dat_crop
    # return np.random.rand(200, 500, 1)


def array2qt(img):
    # From: https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
    h, w, channel = img.shape
    # bytesPerLine = 3 * w
    # return QtGui.QPixmap(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    new_img = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(new_img)
