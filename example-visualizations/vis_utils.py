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
    