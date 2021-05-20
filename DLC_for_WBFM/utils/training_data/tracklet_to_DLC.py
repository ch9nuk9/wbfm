import numpy as np
import pandas as pd


def best_tracklet_covering(df, num_frames_needed, num_frames,
                           verbose=0):
    """
    Given a partially tracked video, choose a series of frames with enough
    tracklets, to be saved in DLC format

    Loops through windows and tracklets, to see if ALL window frames are in the tracklet
    i.e. properly rejects tracklets that skip frames
    """

    def make_window(start_frame):
        return list(range(start_frame,start_frame+num_frames_needed+1))

    x = list(range(num_frames-num_frames_needed))
    y = np.zeros_like(x)
    for i in x:
        which_frames = make_window(i)
        def check_for_full_covering(vals, which_frames=which_frames):
            vals = set(vals)
            return all([f in vals for f in which_frames])

        tracklets_that_cover = df['slice_ind'].apply(check_for_full_covering)
        y[i] = tracklets_that_cover.sum(axis=0)

    best_covering = np.argmax(y)
    if verbose >= 1:
        print(f"Best covering starts at volume {best_covering} with {np.max(y)} tracklets")

    return make_window(best_covering), y
