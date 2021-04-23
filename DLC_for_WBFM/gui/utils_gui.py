from PyQt5 import QtGui
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords
import numpy as np


def get_cropped_frame(fname, t, zxy, crop_sz):
    # crop_coords = get_crop_coords(zxy)

    return np.random.rand(200, 500, 1)


def array2qt(img):
    # From: https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
    h, w, channel = img.shape
    # bytesPerLine = 3 * w
    # return QtGui.QPixmap(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    new_img = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(new_img)
