import os
import time
import warnings
from itertools import product

import cv2
# from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from ipywidgets import interact
from matplotlib import transforms
from matplotlib.ticker import NullFormatter
from scipy import ndimage as ndi

from DLC_for_WBFM.utils.postprocessing.base_DLC_utils import xy_from_dlc_dat
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import *


##
## Background subtraction
##

def subtract_background_4d(video, sz=None):
    #     backSub = cv.createBackgroundSubtractorKNN()
    if sz is None:
        sz = (np.array(video[0, 0, ...].shape) / 4).astype('int')
        sz = tuple(sz)

    print("Subtracting background...")
    for t, z in product(range(video.shape[0]), range(video.shape[1])):
        frame = video[t, z, :, :]
        #         video[t,z,:,:] = frame - cv2.blur(frame, sz)
        video[t, z, :, :] = frame - np.mean(frame)
    print("Done")

    return video


##
## Building cropped videos
##

def get_crop_from_ometiff(fname, this_xy, which_z, num_frames, crop_sz=(28, 28, 10), sz_4d=(100, 39)):
    """
    There is a lot of switching with 'xy' and the rows and columns of the video

    Reads the entire file into memory...

    Input
    ----------
    fname : str
        File name of the input video (.ome.tiff)

    this_xy : array
        DLC output of the neuron positions

    which_z : int
        Which slice to center the crop around

    num_frames : int
        Total number of frames to read

    crop_sz : tuple=(28,28,10)
        XYZ size of cube to output

    sz_4d : tuple=(100,39)
        If the video doesn't retain z metadata, uses this to reshape
    """

    # Pre-allocate in proper size for future
    cropped_dat = np.zeros(crop_sz + (num_frames,))
    all_dat = []

    print("Reading video...")
    video = tifffile.imread(fname)
    print(f"Read video of shape {video.shape}")

    #     video = subtract_background_4d(video)

    if len(video.shape) == 3:
        print("Found 2+1d video; was expecting 3+1d. Attempting to reshape using sz_4d...")
        video = np.reshape(video, sz_4d + video.shape[1:])
        print("Successfully reshaped.")
    #     elif len(video.shape) == 4:
    # Format is already TZXY

    tmp = video.shape[1:]
    video_sz_yxz = (tmp[1], tmp[2], tmp[0])
    video_sz_xyz = (tmp[2], tmp[1], tmp[0])

    for i in range(num_frames):

        xyz = np.append(this_xy[i], which_z)
        print(f"Reading frame {i}/{num_frames - 1} at position {xyz}")
        x_ind, y_ind, z_ind = get_crop_coords3d(xyz, crop_sz=crop_sz, clip_sz=video_sz_xyz)
        tmp = np.transpose(video[i, :, :, :][z_ind, :, :][:, y_ind, :][:, :, x_ind], axes=(2, 1, 0))
        if tmp.shape == crop_sz:
            cropped_dat[:, :, :, i] = tmp
        else:
            print(f"Skipping frame {i}; too close to edge")
            print(f"Was size {tmp.shape}; should be size {crop_sz}")
            # keep as zeros

    return cropped_dat


def get_crop_from_ometiff_virtual(fname, this_xy, this_prob,
                                  which_z, num_frames,
                                  crop_sz=(28, 28, 10),
                                  num_slices=None,
                                  flip_x=False,
                                  start_volume=0,
                                  prob_threshold=0.4,
                                  alpha=1.0,
                                  actually_create=True,
                                  actually_crop=True,
                                  verbose=1):
    """
    Reads in a subset of a very large .btf file.

    Equivalent to get_crop_from_ometiff() but refactored so that the entire file
    is not read at once (which is usually ~180 GB)

    Input
    ----------
    fname : str
        File name of the input video (.ome.tiff)

    this_xy : array
        DLC output of the neuron positions

    this_prob : array
        DLC output of the tracking probabilities (confidence)

    which_z : int
        Which slice to center the crop around

    num_frames : int
        Total number of frames to read

    crop_sz : tuple=(28,28,10)
        XYZ size of cube to output

    num_slices : int
        If the video doesn't retain z metadata, uses this to reshape

    flip_x : bool
        Whether to flip the video in x; currently gcamp and mcherry are mirrored

    start_volume : int
        Number of volumes to skip at the beginning

    prob_threshold : float
        Threshold below which to just skip the plane, i.e. set it to zeros

    alpha : float
        Multiplicative factor; needed if the original .btf needs format conversion
        e.g. uint16 -> uint8

    actually_create : bool
        Debug variable; if false, the file is not read

    actually_crop : bool
        Debug variable; if false, crop_sz is ignored

    verbose : int
        Verbosity (0,1,2)

    Input-Output:
        ome.tiff -> np.array
    """

    def update_ind(i, crop_sz):
        """Translate DLC coordinates to crop coordinates"""
        center = this_xy[i]

        x_ind, y_ind = get_crop_coords(center, sz=crop_sz[0:2])
        return x_ind, y_ind

    def build_sz_vars(crop_sz):
        """Translate crop coordinates to image variables"""
        # Convert crop_sz to list for format compatibility
        if actually_crop:
            start_of_each_frame = int(np.floor(which_z - crop_sz[2] / 2))
            end_of_each_frame = int(np.floor(which_z + crop_sz[2] / 2))
        else:
            start_of_each_frame = 0
            end_of_each_frame = num_slices
        which_slices = list(range(start_of_each_frame, end_of_each_frame))
        end_of_each_frame = end_of_each_frame - 1

        frame_height, frame_width = crop_sz[0:2]
        # Format: TZYX
        full_sz = (num_frames, len(which_slices), frame_width, frame_height)
        final_cropped_video = np.zeros(full_sz)

        if start_of_each_frame < 5 and verbose >= 1:
            warnings.warn("As of 14.10.2020, the first several frames are very bad! Do you really mean to use these?")

        if verbose >= 1:
            print(f'Cropping {len(which_slices)} slices, starting at {start_of_each_frame}')

        return start_of_each_frame, end_of_each_frame, which_slices, \
               final_cropped_video

    start_of_each_frame, end_of_each_frame, which_slices, \
    final_cropped_video = build_sz_vars(crop_sz)

    # Initialize time index and tracking location
    start_volume = start_volume * num_slices
    i_rel_volume = 0

    with tifffile.TiffFile(fname, multifile=False) as tif:
        for i, page in enumerate(tif.pages):
            if i == 0:
                full_sz = page.asarray().shape
                if not actually_crop:
                    full_sz = (num_frames, num_slices,) + full_sz
                    final_cropped_video = np.zeros(full_sz)
                if verbose >= 1:
                    print(f"Full size read as {full_sz}")
                x_ind, y_ind = update_ind(i_rel_volume, crop_sz)
            # Set up relative indices
            this_abs_slice = i % num_slices
            this_rel_slice = this_abs_slice - start_of_each_frame
            # Align start of annotations and .btf
            if i < start_volume or this_abs_slice not in which_slices:
                continue
            # Skip if tracking is below confidence
            if this_prob[i_rel_volume] > prob_threshold:
                if verbose >= 2:
                    print(
                        f'Page {i}/{num_frames * num_slices}; volume {i_rel_volume}/{num_frames} to cropped array slice {this_rel_slice}')

                tmp = (alpha * page.asarray()).astype('uint8')
                if flip_x:
                    tmp = np.flip(tmp, axis=1)

                if actually_crop:
                    final_cropped_video[i_rel_volume, this_rel_slice, ...] = tmp[:, x_ind][y_ind]
                else:
                    final_cropped_video[i_rel_volume, this_rel_slice, ...] = tmp

            # Update time index and tracking location
            if this_abs_slice == end_of_each_frame:
                i_rel_volume += 1
                if num_frames is not None and i_rel_volume >= num_frames: break
                x_ind, y_ind = update_ind(i_rel_volume, crop_sz)

    return final_cropped_video


##
## Finding local maxima
##

def local_maxima_3D(data, order=1):
    """From: https://stackoverflow.com/questions/55453110/how-to-find-local-maxima-of-3d-array-in-python

    Detects local maxima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values


def local_maxima_2D(data):
    """
    Data should be 3d, XYT
    """
    t = np.shape(data)[2]
    coords = np.zeros((t, 1))
    values = np.zeros((t, 1))

    for i in range(t):
        coords[i] = np.argmax(data[..., i])
        values[i] = np.amax(data[..., i])

    return coords, values


##
## Finding maxima via heuristics
##

def mean_of_top_percentile(data, percentile=10):
    """
    Data should be 3d, XYT
    """
    t = np.shape(data)[2]
    values = np.zeros((t, 1))

    for i in range(t):
        tmp = data[..., i]
        thresh = np.percentile(tmp, percentile)
        values[i] = np.mean(tmp[np.where(tmp > thresh)])

    return values


##
## Save a matplotlib animation
##
def save_video4d(file, video4d, fontsize=20):
    """Based on: dNMF.py

    Takes a 4d video with t in the last dimension, animates it and saves
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.max(video4d[:, :, :, 0], axis=2))

    time_text = fig.text(0.5, 0.03, 'Frame = 0', horizontalalignment='center', verticalalignment='top',
                         fontsize=fontsize)

    ax.axis('off')

    #     scalebar = ScaleBar(self.scale[0],'um')
    #     ax.add_artist(scalebar)

    def init():
        im.set_data(np.max(video4d[:, :, :, 0], axis=2))
        return (im,)

    def animate(t):
        data_slice = np.max(video4d[:, :, :, t], axis=2)
        im.set_data(data_slice)

        time_text.set_text('Frame = ' + str(t))

        return (im,)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=video4d.shape[3], interval=200, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(file + '-raw.mp4', writer=writer)

    plt.close('all')

    return anim
