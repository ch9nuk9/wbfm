import deeplabcut
from deeplabcut.utils.make_labeled_video import CreateVideo
from deeplabcut.utils.video_processor import VideoProcessorCV as vp
import numpy as np
import pandas as pd
from tqdm import tqdm

##
## Helper functions for converting annotations
##

def build_dlc_annotation_one_tracklet(row,
                                      all_bodyparts,
                                      num_frames=1000,
                                      coord_names=['x','y','likelihood'],
                                      min_length=5,
                                      neuron_ind=1,
                                      verbose=0):
    # TODO: Check z

    # Variables to be written
    scorer = 'feature_tracker'
    all_frames = list(range(num_frames))

    tracklet_length = len(row['all_xyz'])
    if tracklet_length < min_length:
        return None

    if verbose >= 1:
        print(f"Found tracklet of length {tracklet_length}")

    # Build a dataframe for one neuron across all frames
    # Will be zeros if not detected in a given frame
    coords = np.zeros((num_frames,len(coord_names),))
    for this_slice, this_xyz, this_prob in zip(row['slice_ind'], row['all_xyz'], row['all_prob']):
        # TODO: only works for xy; this_xyz is format ZXY
        coords[this_slice,0] = this_xyz[1]
        coords[this_slice,1] = this_xyz[2]
        try:
            coords[this_slice,-1] = this_prob
        except:
            coords[this_slice,-1] = 0.0
            pass

    index = pd.MultiIndex.from_product([[scorer], [f'neuron{neuron_ind}'],
                                        coord_names],
                                        names=['scorer', 'bodyparts', 'coords'])
    frame = pd.DataFrame(coords, columns = index, index = all_frames)

    return frame


def build_dlc_annotation_all(clust_df, min_length, num_frames=1000, verbose=0):
    new_dlc_df = None
    all_bodyparts = np.asarray(clust_df['clust_ind'])

    neuron_ind = 1
    for i, row in tqdm(clust_df.iterrows(), total=clust_df.shape[0]):
        opt = {'min_length':min_length, 'verbose':verbose-1,
               'neuron_ind':neuron_ind,
               'num_frames':num_frames}
        frame = build_dlc_annotation_one_tracklet(row, all_bodyparts, **opt)
        if frame is not None:
            new_dlc_df = pd.concat([new_dlc_df, frame],axis=1)
            neuron_ind = neuron_ind + 1

#         if verbose >= 1:
#             print("============================")
#             print(f"Row {i}/{len(all_bodyparts)}")

    return new_dlc_df


##
## Actually making the video
##

def make_labeled_video_custom_annotations(dlc_config,
                                          video_fname,
                                          df):
    """
    Wrapper around the deeplabcut video creation functions to work with custom
    annotations
    """

    videooutname = video_fname.replace('.avi', '_labeled.mp4')
    codec="mp4v"
    clip = vp(fname=video_fname, sname=videooutname, codec=codec)
    cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_config)

    displayedbodyparts="all"
    bodyparts = deeplabcut.auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
            cfg, displayedbodyparts
        )
    labeled_bpts = [
        bp
        for bp in df.columns.get_level_values("bodyparts").unique()
        if bp in bodyparts
    ]

    trailpoints = 0

    cropping = False
    [x1, x2, y1, y2] = [0,0,0,0]

    bodyparts2connect = False
    draw_skeleton = False
    skeleton_color = None
    displaycropped = False
    color_by = "bodypart"

    # Actual function call
    CreateVideo(clip,
                df,
                0.0, # Do not remove neurons for confidence
                cfg["dotsize"],
                cfg["colormap"],
                labeled_bpts,
                trailpoints,
                cropping,
                x1,
                x2,
                y1,
                y2,
                bodyparts2connect,
                skeleton_color,
                draw_skeleton,
                displaycropped,
                color_by
            )


##
## Full pipeline: from dataframe to video
##

def make_labeled_video_from_dataframe(config):
    """
    See make_labeled_video_custom_annotations()
    """
    return
