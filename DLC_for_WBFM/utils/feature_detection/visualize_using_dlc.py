import deeplabcut
from deeplabcut.utils.make_labeled_video import CreateVideo
from deeplabcut.utils.video_processor import VideoProcessorCV as vp
from DLC_for_WBFM.bin.configuration_definition import *
from DLC_for_WBFM.utils.preprocessing.convert_matlab_annotations_to_DLC import csv_annotations2config_names
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

    # videooutname = video_fname.replace('.avi', '_labeled.mp4')
    # codec="mp4v"
    videooutname = video_fname.replace('.avi', '_labeled.avi')
    codec = 'JPEG'
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

def create_video_from_annotations(config, df_fname,
                                  scorer=None,
                                  min_track_length=50,
                                  total_num_frames=500):
    """
    Creates a video starting from a saved dataframe of tracklets
    """

    c = load_config(config)
    project_folder = c.get_dirname()

    # Load the dataframe name, and produce DLC-style annotations
    with open(df_fname, 'rb') as f:
        clust_df = pickle.load(f)
    new_dlc_df = build_dlc_annotation_all(clust_df, min_length=min_track_length, num_frames=total_num_frames, verbose=0)

    # Save using DLC-style names
    if scorer is None:
        # Assume the df filename is something like blah_blah_blah_informative.pickle
        scorer_base = df_fname.split('.')[0] # get rid of extension
        scorer = scorer_base.split('_')[-1]
        scorer = f"feature_tracker_{scorer}"

    def build_dlc_name(ext):
        return os.path.join(project_folder,"CollectedData_" + scorer + ext)
    with open(build_dlc_name(".csv"), 'w') as f:
        new_dlc_df.to_csv(f)
#     with open(build_dlc_name(".h5"), 'wb') as f:
#         new_dlc_df.to_hdf(f, key="df_with_missing", mode="w")
    new_dlc_df.to_hdf(build_dlc_name(".h5"), key="df_with_missing", mode="w")
    with open(build_dlc_name(".pickle"), 'wb') as f:
        new_dlc_df.to_pickle(f)

    # Add these annotations to the config file
    # Assume the dlc project is initialized propery
    dlc_config = c.tracking.DLC_config_fname

    annotation_fname = build_dlc_name(".h5")
    annotation_fname = os.path.join(project_folder, annotation_fname)

    vid_fname = c.datafiles.red_avi_fname
    tracking = DLCForWBFMTracking(dlc_config, vid_fname, annotation_fname)

    c.tracking = tracking
    save_config(c)

    # Synchronize the DLC config file
    # Assumes the DLC project is already made
    csv_annotations = c.tracking.annotation_fname.replace('h5', 'csv')
    csv_annotations2config_names(dlc_config, csv_annotations)

    # Finally, make the video
    make_labeled_video_custom_annotations(dlc_config, vid_fname, new_dlc_df)

    return new_dlc_df
