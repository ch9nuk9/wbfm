import os
from math import ceil
import numpy as np


def get_crop_coords(center, sz=(28, 28)):
    x_ind = range(int(center[0] - sz[0] / 2), int(center[0] + sz[0] / 2))
    y_ind = range(int(center[1] - sz[1] / 2), int(center[1] + sz[1] / 2))
    return list(x_ind), list(y_ind)


def get_crop_from_avi(fname, this_xy, num_frames, sz=(28, 28), start_frame=0):
    """
    Gets np.array from .avi video

    Note: reads in entire array into memory

    Output format is TZXY (Z is 1d)
    """
    import cv2  # Sometimes cv2 is a pain to install

    if not os.path.isfile(fname):
        raise ValueError

    cap = cv2.VideoCapture(fname)

    # Pre-allocate in proper size for future
    cropped_dat = np.zeros((num_frames, 1) + sz)
    all_dat = []

    for i in range(num_frames):
        if i < start_frame:
            continue
        ret, frame = cap.read()

        x_ind, y_ind = get_crop_coords(this_xy[i], sz)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            cropped = gray[:, x_ind][y_ind]
            cropped_dat[i - start_frame, 0, :, :] = cropped
        except IndexError:
            continue

    cap.release()

    return cropped_dat


def get_crop_coords3d(center, crop_sz=(28, 28, 10), clip_sz=None):
    x_ind = range(ceil(center[0] - crop_sz[0] / 2), int(center[0] + crop_sz[0] / 2) + 1)
    y_ind = range(ceil(center[1] - crop_sz[1] / 2), int(center[1] + crop_sz[1] / 2) + 1)
    z_ind = range(ceil(center[2] - crop_sz[2] / 2), int(center[2] + crop_sz[2] / 2) + 1)
    if clip_sz is not None:
        x_ind = np.clip(x_ind, 0, clip_sz[0] - 1)
        y_ind = np.clip(y_ind, 0, clip_sz[1] - 1)
        z_ind = np.clip(z_ind, 0, clip_sz[2] - 1)
    return np.array(x_ind), np.array(y_ind), np.array(z_ind)
