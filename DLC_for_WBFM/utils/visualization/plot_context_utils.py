import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.patches as patches


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
