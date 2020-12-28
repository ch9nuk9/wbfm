import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.patches as patches
import imageio
import pickle

from DLC_for_WBFM.utils.visualization.plot_traces import get_tracking_channel, get_measurement_channel, set_big_font
from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import get_crop_from_ometiff_virtual
from DLC_for_WBFM.utils.postprocessing.coconfig_cropping_utils import _get_crop_from_ometiff_virtual
from DLC_for_WBFM.bin.configuration_definition import *
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import *
from DLC_for_WBFM.utils.postprocessing.base_DLC_utils import xy_from_dlc_dat

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
                           trace_data=None,
                           which_field='ratio'):
    """
    Convenience function for plotting using a config file directly

    See also: plot_video_crop_trace
    """

    config = load_config(config_file)

    # Read traces
    if trace_data is None:
        trace_data = pickle.load(open(config.traces.traces_fname, 'rb'))
        try:
            trace_data = np.array(trace_data[which_neuron][which_field])
        except:
            if which_field=='ratio':
                r = get_tracking_channel(trace_data[which_neuron])
                g = get_measurement_channel(trace_data[which_neuron])
                trace_data = g / r

    if video_data is None:
        video_reader = imageio.get_reader(config.tracking.labeled_video_fname)
        video_data = []
        for im in video_reader:
            video_data.append(im)

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
                                    trace_data)
    args = {'t':(0,num_frames-1), 'z':(0,crop_sz[-1]-1)}

    return interact(f, **args)


def plot_video_crop_trace_frame(t, z, video_dat,
                                cropped_dat_mcherry,
                                cropped_dat_gcamp,
                                trace_data):
    """
    Plots a single frame of a video, cropped data, and the trace
        Made to be used as a subfunction of plot_video_crop_trace

    See also: plot_video_crop_trace
    """

    # plt.figure()
    fig = plt.figure(figsize=(45,15))
    # specs = fig.add_gridspec(ncols=1,nrows=3, height_ratios=[10,5,1])

    # 2d video; no z component
    # ax = fig.add_subplot(specs[0])
    # plt.subplot(311)
    plt.imshow(video_dat[t])
    # ax.imshow(video_dat[t])
    plt.title('Full video')

    fig = plt.figure(figsize=(45,15))
    # 3d crop; t and z
    plt.subplot(323)
    plt.imshow(cropped_dat_mcherry[t,z,...])
    plt.clim([0,0.5*np.max(cropped_dat_mcherry)])
    plt.title('Cropped neuron (red)')
    plt.colorbar()
    plt.subplot(324)
    plt.imshow(cropped_dat_gcamp[t,z,...])
    plt.clim([0,1.0*np.max(cropped_dat_gcamp[t,...])])
    plt.title('Cropped neuron (green)')
    plt.colorbar()

    # Trace: all with a line
    plt.subplot(313)
    plt.plot(trace_data)
    plt.vlines(t,0,np.max(np.array(trace_data)), colors='r')
    plt.title('Trace')
    plt.ylim([0,2])

    set_big_font()

    # plt.show()
