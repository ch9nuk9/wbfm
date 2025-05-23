import matplotlib.pyplot as plt
import numpy as np

from wbfm.utils.general.postprocessing.base_cropping_utils import get_crop_from_avi
from wbfm.utils.visualization.utils_plot_traces import set_big_font


##
## Functions for visualizing how the tracking went
##

def interact_box_around_track(video_fname_mcherry,
                              video_fname_gcamp,
                              cropped_dat_mcherry=None,
                              cropped_dat_gcamp=None,
                              this_xy=None,
                              num_frames=100,
                              crop_sz=(19, 19)
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
        plt.figure(figsize=(15, 5))
        _, ax1 = plt.subplots(1)
        plt.imshow(cropped_dat_mcherry[:, :, 0, i]);
        plt.title('mcherry')
        plt.colorbar()
        plt.clim(0, 200)

        _, ax2 = plt.subplots(2)
        plt.imshow(cropped_dat_gcamp[:, :, 0, i]);
        plt.title('gcamp')
        plt.colorbar()


##
## Function for syncing videos and traces
##
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
    fig = plt.figure(figsize=(25, 5))
    # specs = fig.add_gridspec(ncols=1,nrows=3, height_ratios=[10,5,1])

    # 2d video; no z component
    # ax = fig.add_subplot(specs[0])
    # plt.subplot(311)
    plt.imshow(video_dat[t])
    # ax.imshow(video_dat[t])
    plt.title('Full video')

    fig = plt.figure(figsize=(45, 15))
    # 3d crop; t and z
    plt.subplot(323)
    plt.imshow(cropped_dat_mcherry[t, z, ...])
    plt.clim([0, 0.5 * np.max(cropped_dat_mcherry)])
    plt.title('Cropped neuron (red)')
    plt.colorbar()
    plt.subplot(324)
    plt.imshow(cropped_dat_gcamp[t, z, ...])
    plt.clim([0, 1.0 * np.max(cropped_dat_gcamp[t, ...])])
    plt.title('Cropped neuron (green)')
    plt.colorbar()

    # Trace: all with a line
    plt.subplot(313)
    plt.plot(trace_data)
    # plt.vlines(t,0,np.max(np.array(trace_data)), colors='r')
    plt.vlines(t, 0, 2, colors='r')
    plt.title('Trace')
    plt.ylim([0, 2])

    set_big_font()

    # plt.show()
