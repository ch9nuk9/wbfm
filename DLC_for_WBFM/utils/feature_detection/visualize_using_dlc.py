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

    if verbose >= 2:
        print(f"Found tracklet of length {tracklet_length}")
    if tracklet_length < min_length:
        return None

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


def build_dlc_annotation_all(clust_df, min_length, num_frames=1000, verbose=1):
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

    if verbose >= 1 and new_dlc_df is not None:
        print(f"Found {len(new_dlc_df.columns)/3} tracks of length >{min_length}")

    return new_dlc_df


##
## Actually making the video
##

def make_labeled_video_custom_annotations(dlc_config,
                                          video_fname,
                                          df,
                                          video_suffix=''):
    """
    Wrapper around the deeplabcut video creation functions to work with custom
    annotations
    """

    # videooutname = video_fname.replace('.avi', '_labeled.mp4')
    # codec="mp4v"
    videooutname = video_fname.replace('.avi', f'_labeled-{video_suffix}.avi')
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
## Utilities for saving intermediate DLC products
##

def save_dlc_annotations(scorer, df_fname, c, new_dlc_df):

    project_folder = c.get_dirname()

    # Save using DLC-style names
    if scorer is None:
        # Assume the df filename is something like blah_blah_blah_informative.pickle
        scorer_base = df_fname.split('.')[0] # get rid of extension
        scorer = scorer_base.split('_')[-1]
        scorer = f"feature_tracker_{scorer}"

    # Build the filenames that will be written
    def build_dlc_name(ext):
        return os.path.join(project_folder,"CollectedData_" + scorer + ext)
    all_ext = [".csv", ".h5", ".pickle"]
    all_fnames = [build_dlc_name(ext) for ext in all_ext]

    with open(all_fnames[0], 'w') as f:
        new_dlc_df.to_csv(f)
    new_dlc_df.to_hdf(all_fnames[1], key="df_with_missing", mode="w")
    with open(all_fnames[2], 'wb') as f:
        new_dlc_df.to_pickle(f)

    return build_dlc_name, all_fnames


def synchronize_config_files(c, build_dlc_name):
    """Synchronizes my config file and the DLC config file"""

    project_folder = c.get_dirname()
    # Add these annotations to the config file
    # Assume the dlc project is initialized properly
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

    # Load the dataframe name, and produce DLC-style annotations
    with open(df_fname, 'rb') as f:
        clust_df = pickle.load(f)
    opt = {'min_length':min_track_length, 'num_frames':total_num_frames, 'verbose':1}
    new_dlc_df = build_dlc_annotation_all(clust_df, **opt)
    if new_dlc_df is None:
        print("Found no tracks long enough; aborting")
        return None

    # Save annotations using DLC-style names, and update the config files
    build_dlc_name = save_dlc_annotations(scorer, df_fname, c, new_dlc_df)[0]
    synchronize_config_files(c, build_dlc_name)

    # Finally, make the video
    dlc_config = c.tracking.DLC_config_fname
    vid_fname = c.datafiles.red_avi_fname
    make_labeled_video_custom_annotations(dlc_config, vid_fname, new_dlc_df)

    return new_dlc_df


def create_many_videos_from_annotations(config, df_fname,
                                      min_track_length=400,
                                      total_num_frames=500,
                                      num_videos_to_make=None,
                                      verbose=1):
    """
    Creates a large number of videos from a single annotation dataframe
        One neuron per video, for ease of manual error checking

    See create_video_from_annotations()

    """
    c = load_config(config)
    project_folder = c.get_dirname()

    # Load the dataframe name, and produce DLC-style annotations
    with open(df_fname, 'rb') as f:
        clust_df = pickle.load(f)
    opt = {'min_length':min_track_length, 'num_frames':total_num_frames, 'verbose':0}
    # Loop through tracklets, and make a video for each
    all_bodyparts = np.asarray(clust_df['clust_ind'])

    neuron_ind = 1 # Only for video indices
    opt['neuron_ind'] = 1
    for _, row in clust_df.iterrows():
        this_dlc_df = build_dlc_annotation_one_tracklet(row, all_bodyparts, **opt)
        if this_dlc_df is None:
            continue
        else:
            neuron_ind = neuron_ind + 1

        # Check the folder to build a unique name for the scorer
        scorer = f'neuron-{neuron_ind-1}'
        _, _, filenames = next(os.walk(project_folder))
        tmp_files = [f for f in filenames if scorer in f]
        if len(tmp_files)>0:
            print(f"Temporary files found; delete or move them:")
            print(tmp_files)
            print("ABORTING")
            return

        # Save annotations using DLC-style names, and update the config files
        build_dlc_name, prev_fnames = save_dlc_annotations(scorer, df_fname, c, this_dlc_df)
        synchronize_config_files(c, build_dlc_name)

        # Finally, make the video
        dlc_config = c.tracking.DLC_config_fname
        vid_fname = c.datafiles.red_avi_fname
        make_labeled_video_custom_annotations(dlc_config, vid_fname, this_dlc_df,
                                              video_suffix=scorer)

        # Clean up by deleting the temporary .h5/.csv/.pickle files
        [os.remove(f) for f in prev_fnames]
        if (num_videos_to_make is not None) and (neuron_ind > num_videos_to_make):
            break

    if verbose >= 1:
        print(f"Finished making videos for {neuron_ind-1} neurons")
