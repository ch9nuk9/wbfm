import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.patches as patches
import imageio
import pickle

from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import xy_from_dlc_dat, get_crop_from_ometiff_virtual, _get_crop_from_ometiff_virtual
from DLC_for_WBFM.bin.configuration_definition import *

##
## Functions for visualizing how the tracking went
##

def interact_box_around_track(video_fname_mcherry,
                              video_fname_gcamp,
                              cropped_dat_mcherry=None,
                              cropped_dat_gcamp=None,
                              this_xy=None,
                              num_frames=100,
                              crop_sz=(19,19)
                              ):
    """
    Takes a .avi video and tracks, and produces a widget

    Can also pass the data directly

    UNFINISHED
    """

    if cropped_dat_mcherry is None:
        cropped_dat_mcherry = get_crop_from_avi(video_fname_mcherry, this_xy, num_frames, sz=crop_sz)
    if cropped_dat_gcamp is None:
        cropped_dat_gcamp = get_crop_from_avi(video_fname_gcamp, this_xy, num_frames, sz=crop_sz)

    def f(i):
        # Get frame for current time
        plt.figure(figsize=(15,5))
        _,ax1 = plt.subplots(1)
        plt.imshow(cropped_dat_mcherry[:,:,0,i]);
        plt.title('mcherry')
        plt.colorbar()
        plt.clim(0, 200)

        _,ax2 = plt.subplots(2)
        plt.imshow(cropped_dat_gcamp[:,:,0,i]);
        plt.title('gcamp')
        plt.colorbar()



##
## Function for syncing videos and traces
##


def _plot_video_crop_trace(config_file,
                           which_neuron,
                           num_frames,
                           video_data=None,
                           green_data=None,
                           red_data=None,
                           trace_data=None):
    """
    Convenience function for plotting using a config file directly

    See also: plot_video_crop_trace
    """

    config = load_config(config_file)

    # Read traces
    if trace_data is None:
        trace_data = pickle.load(open(config.traces.traces_fname, 'rb'))
        trace_data = np.array(trace_dat[which_neuron][which_field])

    if video_data is None:
        video_reader = imageio.get_reader(config.datafiles.red_avi_fname)
        video_dat = []
        for im in video_reader:
            video_dat.append(im)

    if green_data is None:
        green_data = _get_crop_from_ometiff_virtual(config,
                                                    which_neuron,
                                                    num_frames,
                                                    use_red_channel=False)
    if red_data is None:
        red_data = _get_crop_from_ometiff_virtual(config,
                                                  which_neuron,
                                                  num_frames)

    # Widget for interaction
    crop_sz = config.traces.crop_sz

    f = lambda t,z : \
        plot_video_crop_trace_frame(t, z,
                                    video_data,
                                    red_data,
                                    green_data,
                                    trace_data)
    args = {'t':(0,num_frames-1), 'z':(0,crop_sz[-1]-1)}

    return interact(f, **args)



def plot_video_crop_trace(vid_fname,
                          gcamp_fname,
                          mcherry_fname,
                          annotation_fname,
                          trace_fname,
                          which_neuron,
                          num_frames,
                          crop_sz,
                          which_z,
                          which_field='gcamp',
                          num_slices=33,
                          alpha=1.0,
                          flip_x=False,
                          start_volume=0):
    """
    Plots in 3 panels:
        - Tracked video or behavior
        - Cropped data
        - Trace
    """

    # Read in video
    video_reader = imageio.get_reader(vid_fname)
    video_dat = []
    for im in video_reader:
        video_dat.append(im)

    # Read in crop
    cropped_dat = []
    this_xy, this_prob = xy_from_dlc_dat(annotation_fname,
                                         which_neuron=which_neuron,
                                         num_frames=num_frames)
    cropped_dat_mcherry = get_crop_from_ometiff_virtual(mcherry_fname,
                                                this_xy,
                                                which_z,
                                                num_frames,
                                                crop_sz=crop_sz,
                                                num_slices=num_slices,
                                                alpha=alpha,
                                                flip_x= ~flip_x,
                                                start_volume=start_volume,
                                                verbose=False)
    cropped_dat_gcamp = get_crop_from_ometiff_virtual(gcamp_fname,
                                               this_xy,
                                               which_z,
                                               num_frames,
                                               crop_sz=crop_sz,
                                               num_slices=num_slices,
                                               alpha=alpha,
                                               flip_x=flip_x,
                                               start_volume=start_volume,
                                               verbose=False)


    # Read traces
    trace_dat = pickle.load(open(trace_fname, 'rb'))
    trace_dat = np.array(trace_dat[which_neuron][which_field])

    # Widget for interaction
    f = lambda t,z : \
        plot_video_crop_trace_frame(t, z, video_dat,
                                    cropped_dat_mcherry,
                                    cropped_dat_gcamp,
                                    trace_dat)
    args = {'t':(0,num_frames-1), 'z':(0,crop_sz[-1]-1)}

    return interact(f, **args)


def plot_video_crop_trace_frame(t, z, video_dat,
                                cropped_dat_mcherry,
                                cropped_dat_gcamp,
                                trace_dat):
    """
    Plots a single frame of a video, cropped data, and the trace
        Made to be used as a subfunction of plot_video_crop_trace

    See also: plot_video_crop_trace
    """

    # plt.figure()
    plt.figure(figsize=(45,15))

    # 2d video; no z component
    plt.subplot(231)
    plt.imshow(video_dat[t])
    plt.title('Full video')

    # 3d crop; t and z
    plt.subplot(232)
    plt.imshow(cropped_dat_mcherry[t,z,...])
    plt.clim([0,0.5*np.max(cropped_dat_mcherry[t,...])])
    plt.title('Cropped neuron (red)')
    plt.colorbar()
    plt.subplot(233)
    plt.imshow(cropped_dat_gcamp[t,z,...])
    plt.clim([0,1.0*np.max(cropped_dat_gcamp[t,...])])
    plt.title('Cropped neuron (green)')
    plt.colorbar()

    # Trace: all with a line
    plt.subplot(212)
    plt.plot(trace_dat)
    plt.vlines(t,0,np.max(trace_dat), colors='r')
    plt.title('Trace')

    # plt.show()
