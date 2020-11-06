import numpy as np
import cv2
import tifffile
import h5py
import os
import matplotlib.pyplot as plt
from ipywidgets import interact
from scipy import ndimage as ndi
from itertools import product
from math import ceil
from matplotlib.ticker import NullFormatter
from matplotlib import transforms
from dNMF.Demix.dNMF import dNMF
import torch
import time

# from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.animation as animation

##
## Background subtraction
##

def subtract_background_4d(video, sz=None):
#     backSub = cv.createBackgroundSubtractorKNN()
    if sz is None:
        sz = (np.array(video[0,0,...].shape)/4).astype('int')
        sz = tuple(sz)

    print("Subtracting background...")
    for t, z in product(range(video.shape[0]), range(video.shape[1])):
        frame = video[t,z,:,:]
#         video[t,z,:,:] = frame - cv2.blur(frame, sz)
        video[t,z,:,:]  = frame - np.mean(frame)
    print("Done")

    return video


##
## Reading DeepLabCut
##

def xy_from_dlc_dat(fname, which_neuron=0, num_frames=100):

    xy_ind = range(which_neuron*3, which_neuron*3 + 2)
    prob_ind = which_neuron*3 + 2

    this_xy = []
    this_prob = []

    with h5py.File(fname, 'r') as dlc_dat:
        dlc_table = dlc_dat['df_with_missing']['table']
        which_frames = range(num_frames)
        this_xy.extend([this_frame[1][xy_ind] for this_frame in dlc_table[which_frames]])
        this_prob.extend([this_frame[1][prob_ind] for this_frame in dlc_table[which_frames]])

    return this_xy, this_prob


##
## Building cropped videos
##

def get_crop_coords(center, sz=(28,28)):
    x_ind = range(int(center[0] - sz[0]/2), int(center[0] + sz[0]/2))
    y_ind = range(int(center[1] - sz[1]/2), int(center[1] + sz[1]/2))
    return list(x_ind), list(y_ind)


def get_crop_from_avi(fname, this_xy, num_frames, sz=(28,28)):

    if not os.path.isfile(fname):
        raise FileException

    cap = cv2.VideoCapture(fname)

    # Pre-allocate in proper size for future
    cropped_dat = np.zeros(sz+(1,num_frames))
    all_dat = []

    for i in range(num_frames):
        ret, frame = cap.read()

        x_ind, y_ind = get_crop_coords(this_xy[i], sz)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            cropped = gray[:,x_ind][y_ind]
            cropped_dat[:,:,0,i] = cropped
        except:
            continue

    cap.release()

    return cropped_dat

def get_crop_coords3d(center, crop_sz=(28,28,10), clip_sz=None):
    x_ind = range(ceil(center[0] - crop_sz[0]/2), int(center[0] + crop_sz[0]/2)+1)
    y_ind = range(ceil(center[1] - crop_sz[1]/2), int(center[1] + crop_sz[1]/2)+1)
    z_ind = range(ceil(center[2] - crop_sz[2]/2), int(center[2] + crop_sz[2]/2)+1)
    if clip_sz is not None:
        x_ind = np.clip(x_ind, 0, clip_sz[0]-1)
        y_ind = np.clip(y_ind, 0, clip_sz[1]-1)
        z_ind = np.clip(z_ind, 0, clip_sz[2]-1)
    return np.array(x_ind), np.array(y_ind), np.array(z_ind)


def get_crop_from_ometiff(fname, this_xy, which_z, num_frames, crop_sz=(28,28,10), sz_4d=(100,39)):
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
    cropped_dat = np.zeros(crop_sz+(num_frames,))
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
        print(f"Reading frame {i}/{num_frames-1} at position {xyz}")
        x_ind, y_ind, z_ind = get_crop_coords3d(xyz, crop_sz=crop_sz, clip_sz=video_sz_xyz)
        tmp = np.transpose(video[i,:,:,:][z_ind,:,:][:, y_ind,:][:,:, x_ind], axes=(2,1,0))
        if tmp.shape == crop_sz:
            cropped_dat[:,:,:,i] = tmp
        else:
            print(f"Skipping frame {i}; too close to edge")
            print(f"Was size {tmp.shape}; should be size {crop_sz}")
            # keep as zeros

    return cropped_dat


def get_crop_from_ometiff_virtual(fname, this_xy, which_z, num_frames,
                                  crop_sz=(28,28,10), num_slices=None,
                                  flip_x=False,
                                  start_volume=0,
                                  alpha=1.0,
                                  actually_create=True):
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

    alpha : float
        Multiplicative factor; needed if the original .btf needs format conversion
        e.g. uint16 -> uint8

    actually_create : bool
        Debug variable; if false, the file is not read


    Input-Output:
        ome.tiff -> np.array
    """

    # Convert crop_sz to list for format compatibility
    start_of_each_frame = int(np.floor(which_z - crop_sz[2]/2))
    end_of_each_frame = int(np.floor(which_z + crop_sz[2]/2))
    which_slices = list(range(start_of_each_frame, end_of_each_frame))
    end_of_each_frame = end_of_each_frame-1

    frame_height, frame_width = crop_sz[0:2]

    if start_of_each_frame < 5:
        warnings.warn("As of 14.10.2020, the first several frames are very bad! Do you really mean to use these?")

    # Initialize time index and tracking location
    start_volume = start_volume * num_slices
    i_rel_volume = 0
    # Format: TZYX
    final_cropped_video = np.zeros((num_frames, len(which_slices), frame_width, frame_height))

    print(f'Cropping {len(which_slices)} slices, starting at {start_of_each_frame}' )

    def update_ind(i):
        center = this_xy[i]
        # if flip_x:
        #     center[0] = full_sz[0] - center[0]
            # center[1] = full_sz[1] - center[1]

        x_ind, y_ind = get_crop_coords(center, sz=crop_sz[0:2])
        return x_ind, y_ind

    with tifffile.TiffFile(fname, multifile=False) as tif:
        for i, page in enumerate(tif.pages):
            this_abs_slice = i % num_slices
            this_rel_slice = this_abs_slice - start_of_each_frame
            if i == 0:
                full_sz = page.asarray().shape
                print(full_sz)
                x_ind, y_ind = update_ind(i_rel_volume)
                # final_cropped_video =  np.zeros((num_frames, len(which_slices), full_sz[0], full_sz[1]))
                # final_cropped_video =  np.zeros((num_frames, len(which_slices), full_sz[0], frame_height))

            # Align start of annotations and .btf
            if i < start_volume or this_abs_slice not in which_slices:
                continue
            print(f'Page {i}/{num_frames*num_slices}; volume {i_rel_volume}/{num_frames} to cropped array slice {this_rel_slice}')

            tmp = (alpha*page.asarray()).astype('uint8')
            if flip_x:
                tmp = np.flip(tmp,axis=1)
            final_cropped_video[i_rel_volume, this_rel_slice,...] = tmp[:,x_ind][y_ind]
            # final_cropped_video[i_rel_volume, this_rel_slice,...] = tmp[:,x_ind]
            # if not flip_x:
            #     final_cropped_video[i_rel_volume, this_rel_slice,...] = tmp[:,x_ind][y_ind]
            # else:
                # full_height = tmp.shape[1]
                # tmp_x_ind = [full_height - i for i in x_ind]
                # final_cropped_video[i_rel_volume, this_rel_slice,...] = (
                #         np.flip(tmp,axis=1))[:,tmp_x_ind][y_ind]
                # full_height = tmp.shape[1]
                # tmp_x_ind = [full_height - i for i in x_ind]
                # final_cropped_video[i_rel_volume, this_rel_slice,...] = (
                #         np.flip(tmp,axis=1))[:,tmp_x_ind][y_ind]

            # Update time index and tracking location
            if this_abs_slice == end_of_each_frame:
                i_rel_volume += 1
                x_ind, y_ind = update_ind(i_rel_volume)

            if num_frames is not None and i_rel_volume >= num_frames: break

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
    coords = np.zeros((t,1))
    values = np.zeros((t,1))

    for i in range(t):
        coords[i] = np.argmax(data[...,i])
        values[i] = np.amax(data[...,i])

    return coords, values


##
## Finding maxima via heuristics
##

def mean_of_top_percentile(data, percentile=10):
    """
    Data should be 3d, XYT
    """
    t = np.shape(data)[2]
    values = np.zeros((t,1))

    for i in range(t):
        tmp = data[...,i]
        thresh = np.percentile(tmp, percentile)
        values[i] = np.mean(tmp[np.where(tmp>thresh)])

    return values


##
## Save a matplotlib animation
##
def save_video4d(file, video4d, fontsize=20):
    """Based on: dNMF.py

    Takes a 4d video with t in the last dimension, animates it and saves
    """
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(np.max(video4d[:,:,:,0],axis=2))

    time_text = fig.text(0.5, 0.03,'Frame = 0',horizontalalignment='center',verticalalignment='top',fontsize=fontsize)

    ax.axis('off')
#     scalebar = ScaleBar(self.scale[0],'um')
#     ax.add_artist(scalebar)

    def init():
        im.set_data(np.max(video4d[:,:,:,0],axis=2))
        return (im,)

    def animate(t):
        data_slice = np.max(video4d[:,:,:,t],axis=2)
        im.set_data(data_slice)

        time_text.set_text('Frame = ' + str(t))

        return (im,)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=video4d.shape[3], interval=200, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(file+'-raw.mp4', writer=writer)

    plt.close('all')

    return anim


##
## Generally plotting
##


def plot2d_with_max(dat, t, max_ind, max_vals, vmin=100, vmax=400):
    plt.imshow(dat[:,:,0,t], vmin=vmin, vmax=vmax)
    plt.colorbar()
    x, y = max_ind[t,1], max_ind[t,0]
    if z == max_ind[t,2]:
        plt.scatter(x, y, marker='x', c='r')
    plt.title(f"Max for t={t} is {max_vals[t]} xy={x},{y}")

def plot3d_with_max(dat, z, t, max_ind, vmin=100, vmax=400):
    plt.imshow(dat[:,:,z,t], vmin=vmin, vmax=vmax)
    plt.colorbar()
    x, y = max_ind[t,1], max_ind[t,0]
    if z == max_ind[t,2]:
        plt.scatter(x, y, marker='x', c='r')
    plt.title(f"Max for t={t} is on z={max_ind[t,2]}, xy={x},{y}")


def plot3d_with_max_and_hist(dat, z, t, max_ind):
    # From: https://matplotlib.org/2.0.2/examples/pylab_examples/scatter_hist.html
    rot = transforms.Affine2D().rotate_deg(90)
    nullfmt = NullFormatter()         # no labels

    plt.figure(1, figsize=(8, 8))


    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axIm = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Actually display
    frame = dat[:,:,z,t]
    axIm.imshow(frame, vmin=0, vmax=400)
    x, y = max_ind[t,1], max_ind[t,0]
#     if z == max_ind[t,2]:
#         plt.scatter(x, y, marker='x', c='r')
#     plt.title(f"Max for t={t} is on z={max_ind[t,2]}, xy={x},{y}")

    axHistx.plot(np.max(frame, axis=0))

#     base = plt.gca().transData
    axHisty.plot(np.flip(np.max(frame, axis=1)), range(frame.shape[0]))#, transform=base+rot)

##
## Functions for use with data from 'extract_all_traces'
##

def visualize_all_traces(all_traces, all_names=None,
                         to_save=False):
    all_names = check_default_names(all_names, len(all_traces))

    for i, t_dict in enumerate(all_traces):
        visualize_mcherry_and_gcamp(t_dict, name=all_names[i])
        if to_save:
            plt.savefig(f'traces_{all_names[i]}')


def visualize_traces_with_reference(all_traces,
                                    reference_ind, reference_name,
                                    all_names=None,
                                    to_normalize=True,
                                    to_save=False):
    """
    Plot all neurons on a reference, given by reference_ind
    """
    all_names = check_default_names(all_names, len(all_traces))

    reference_trace = all_traces[reference_ind]

    for i, t_dict in enumerate(all_traces):
        if i == reference_ind:
            continue
        # Plot looped trace and reference
        ax1, ax2 = visualize_mcherry_and_gcamp(reference_trace, reference_name,
                                               make_new_title=False,
                                               to_normalize=to_normalize)
        visualize_mcherry_and_gcamp(t_dict, all_names[i],
                                    make_new_fig=False,
                                    make_new_title=False,
                                    ax1=ax1, ax2=ax2,
                                    to_normalize=to_normalize)
        if to_save:
            plt.savefig(f'traces_{all_names[i]}_ref_{reference_name}')


def visualize_mcherry_and_gcamp(t_dict, name,
                                make_new_fig=True,
                                make_new_title=True,
                                ax1 = None, ax2 = None,
                                to_normalize=False):
    if make_new_fig:
        plt.figure(figsize=(35,5))#, fontsize=12)

    if make_new_fig:
        ax1 = plt.subplot(121)
    dat = t_dict['mcherry']
    if to_normalize:
        dat = dat / np.max(np.array(dat))
    if make_new_title:
        ax1.plot(dat)
        plt.title(f'mcherry for neuron {name}')
    else:
        ax1.plot(dat, label=f'mcherry for neuron {name}')
        ax1.legend()

    if make_new_fig:
        ax2 = plt.subplot(122)
    dat = t_dict['gcamp']
    if to_normalize:
        dat = dat / np.max(np.array(dat))
    if make_new_title:
        ax2.plot(dat)
        plt.title(f'gcamp for neuron {name}')
    else:
        ax2.plot(dat, label=f'gcamp for neuron {name}')
        ax2.legend()

    return ax1, ax2

def check_default_names(all_names, num_neurons):
    if all_names is None:
        all_names = [str(i) for i in range(num_neurons)]
    return all_names


##
## Full workflow
##

def extract_all_traces(annotation_fname,
                       video_fname_mcherry,
                       video_fname_gcamp,
                       which_neurons=None,
                       num_frames=None,
                       crop_sz=(19,19),
                       params=None):
    """
    Extracts a trace from a single neuron in 2d using dNMF from one movie

    Input
    ----------
    annotation_fname : str
        .h5 produced by DeepLabCut with annotations

    video_fname_mcherry : str
        .avi file with comparison channel.
        As of 16.10.2020 this is 'mcherry'

    video_fname_gcamp : str
        .avi file with actual neuron activities
        As of 16.10.2020 this is 'gcamp'

    which_neuron : [int,..]
        Indices of the neurons, as determined by the original annotation
        By default, extracts all tracked neurons

    num_frames : int
        How many frames to extract

    crop_sz : (int, int)
        Number of pixels to use for traces determination.
        A Gaussian is fit within this size, so it should contain the entire neuron

    params : dict
        Parameters for final trace extraction, using a Gaussian.
        See 'dNMF' docs for explanation of parameters

    Output
    ----------
    all_traces : [dict,...]
        Array of dicts, where the keys are 'mcherry' and 'gcamp'
        Each final element is a 1d array
    """

    # Get the number of neurons
    if which_neurons is None or num_frames is None:
        with h5py.File(annotation_fname, 'r') as dlc_dat:
            dlc_table = dlc_dat['df_with_missing']['table']
            # Each table entry has: x, y, probability
            if which_neurons is None:
                num_neurons = len(dlc_table[0][1])//3
                which_neurons = range(num_neurons)
            if num_frames is None:
                num_frames = len(dlc_table)
        print(f'Found annotations for {num_neurons} neurons and {num_frames} frames')

    # Initialize
    all_traces = []
    start = time.time()

    # Loop through and get traces of gcamp and mcherry
    for which_neuron in which_neurons:
        print(f'Starting analysis of neuron {which_neuron}/{len(which_neurons)}...')
        mcherry_dat = extract_single_trace(annotation_fname,
                                 video_fname_mcherry,
                                 which_neuron=which_neuron,
                                 num_frames=num_frames,
                                 crop_sz=crop_sz,
                                 params=params)
        print('Finished extracting mCherry')
        gcamp_dat = extract_single_trace(annotation_fname,
                                  video_fname_gcamp,
                                  which_neuron=which_neuron,
                                  num_frames=num_frames,
                                  crop_sz=crop_sz,
                                  params=params)
        print('Finished extracting GCaMP')
        all_traces.append({'mcherry':mcherry_dat,
                           'gcamp':gcamp_dat})
    end = time.time()
    print('Finished in ' + str(end-start) + ' seconds')

    return all_traces


def extract_single_trace(annotation_fname,
                         video_fname,
                         which_neuron=0,
                         num_frames=500,
                         crop_sz=(19,19),
                         params=None):
    """
    Extracts a trace from a single neuron in 2d using dNMF from one movie

    Input
    ----------
    annotation_fname : str
        .h5 produced by DeepLabCut with annotations

    video_fname : str
        .avi file with neuron activities.
        Intended use is red or green channel

    which_neuron : int
        Index of the neuron, as determined by the original annotation

    num_frames : int
        How many frames to extract

    crop_sz : (int, int)
        Number of pixels to use for traces determination.
        A Gaussian is fit within this size, so it should contain the entire neuron

    params : dict
        Parameters for final trace extraction, using a Gaussian.
        See 'dNMF' docs for explanation of parameters

    Output
    ----------
    trace : np.array()
        1d array of trace activity
    """

    # Get the positions, and crop the full video
    this_xy, this_prob = xy_from_dlc_dat(annotation_fname,
                                         which_neuron=which_neuron,
                                         num_frames=num_frames)
    cropped_dat = get_crop_from_avi(video_fname,
                                    this_xy,
                                    num_frames,
                                    sz=crop_sz)

    # Get parameters and run dNMF
    dnmf_obj = dNMF_default_from_DLC(cropped_dat, crop_sz, params)
    dnmf_obj.optimize(lr=1e-4,n_iter=20,n_iter_c=2)

    return dnmf_obj.C[0,:]




def dNMF_default_from_DLC(dat, crop_sz, params=None):
    """
    Prepares the parameters and data files for consumption by dNMF
    """
    # Defaults that work decently
    if params is None:
        params = {'n_trials':5, 'noise_level':1e-2, 'sigma_inv':.2,
                  'radius':10, 'step_S':.1, 'gamma':0, 'stride_factor':2, 'density':.1, 'varfact':5,
                  'traj_means':[.0,.0,.0], 'traj_variances':[2e-4,2e-4,1e-5], 'sz':[20,20,1],
                  'K':20, 'T':100, 'roi_window':[4,4,0]}

    # Build position and convert to pytorch
    positions =[list(crop_sz + (0,)),[1, 1, 0]] # Add a dummy position
    positions = np.expand_dims(positions,2)/2.0 # Return the center of the crop
    positions =  torch.tensor(positions).float()

    # Convert the data
    dat_torch = torch.tensor(dat).float()

    # Finalize the parameters
    params = {'positions':positions[:,:,0][:,:,np.newaxis],\
              'radius':params['radius'],'step_S':params['step_S'],'gamma':params['gamma'],\
              'use_gpu':False,'initial_p':positions[:,:,0],'sigma_inv':params['sigma_inv'],\
              'method':'1->t', 'verbose':True, 'use_gpu':False}

    # Finally, create the analysis object
    dnmf_obj = dNMF(dat_torch, params=params)

    return dnmf_obj
