import deeplabcut
from deeplabcut.utils.make_labeled_video import CreateVideo
from deeplabcut.utils.video_processor import VideoProcessorCV as vp
from deeplabcut.utils import auxiliaryfunctions
# from DLC_for_WBFM.config.class_configuration import load_config, DLCForWBFMTracking, save_config
import os
import tifffile
from DLC_for_WBFM.utils.preprocessing.convert_matlab_annotations_to_DLC import csv_annotations2config_names
from DLC_for_WBFM.utils.preprocessing.utils_tif import perform_preprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import build_subset_df_from_tracklets, \
    build_dlc_annotation_one_tracklet, build_dlc_annotation_from_tracklets


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
    bodyparts = set(deeplabcut.auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
            cfg, displayedbodyparts
        ))
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

def save_dlc_annotations(scorer, df_fname, new_dlc_df, project_folder=None, c=None):

    if project_folder is None:
        project_folder = c.get_dirname()

    # Save using DLC-style names
    if scorer is None:
        # Assume the df filename is something like blah_blah_blah_informative.pickle
        scorer_base = df_fname.split('.')[0] # get rid of extension
        scorer = scorer_base.split('_')[-1]
        scorer = f"feature_tracker_{scorer}"

    # Build the filenames that will be written
    def build_dlc_name(ext):
        return os.path.join(project_folder, "CollectedData_" + scorer + ext)
    all_ext = [".csv", ".h5", ".pickle"]
    all_fnames = [build_dlc_name(ext) for ext in all_ext]

    with open(all_fnames[0], 'w') as f:
        new_dlc_df.to_csv(f)
    new_dlc_df.to_hdf(all_fnames[1], key="df_with_missing", mode="w")
    with open(all_fnames[2], 'wb') as f:
        new_dlc_df.to_pickle(f)

    return build_dlc_name, all_fnames


def synchronize_config_files(c, build_dlc_name,
                             dummy_subfolder=None,
                             scorer=None,
                             num_dims=2):
    """Synchronizes my config file and the DLC config file"""

    project_folder = c.get_dirname()
    # Add these annotations to the config file
    # Assume the dlc project is initialized properly
    dlc_config = c.tracking.DLC_config_fname

    annotation_fname = build_dlc_name(".h5")
    annotation_fname = os.path.join(project_folder, annotation_fname)

    video_fname = c.datafiles.red_avi_fname
    tracking = DLCForWBFMTracking(dlc_config, video_fname, annotation_fname)

    c.tracking = tracking
    save_config(c)

    # Synchronize the DLC config file

    # First add the annotated body parts
    h5_annotations = c.tracking.annotation_fname
    csv_annotations2config_names(dlc_config, h5_annotations, num_dims=num_dims+1)

    # Next add the video or shortened video folder name
    cfg = auxiliaryfunctions.read_config(dlc_config)
    nocrop = {"crop": "0, 0, 0, 0"}
    vid_dict = {}
    if dummy_subfolder is None:
        vid_dict[video_fname] = nocrop
    else:
        # Note: do NOT want a filesep here
        dummy_fname = dummy_subfolder+".tif"
        vid_dict[dummy_fname] = nocrop

    cfg["video_sets"] = vid_dict

    # Make sure we are in 3d mode
    cfg["using_z_slices"] = True

    if scorer is not None:
        cfg["scorer"] = scorer

    auxiliaryfunctions.write_config(dlc_config, cfg)


##
## Full pipeline: from dataframe to video
##

def create_video_from_annotations(config, df_fname,
                                  scorer=None,
                                  min_track_length=50,
                                  total_num_frames=500,
                                  coord_names=['x','y','likelihood'],
                                  verbose=0):
    """
    Creates a video starting from a saved dataframe of tracklets
    """

    c = load_config(config)

    # Load the dataframe name, and produce DLC-style annotations
    clust_df = pd.read_pickle(df_fname)
    # with open(df_fname, 'rb') as f:
        # clust_df = pickle.load(f)
    options = {'min_length':min_track_length, 'num_frames':total_num_frames,
           'coord_names':coord_names,
           'verbose':verbose}
    new_dlc_df = build_dlc_annotation_from_tracklets(clust_df, **options)
    if new_dlc_df is None:
        print("Found no tracks long enough; aborting")
        return None

    # Save annotations using DLC-style names, and update the config files
    build_dlc_name = save_dlc_annotations(scorer, df_fname, new_dlc_df, c=c)[0]
    synchronize_config_files(c, build_dlc_name, num_dims=len(coord_names)-1)

    # Finally, make the video
    dlc_config = c.tracking.DLC_config_fname
    video_fname = c.datafiles.red_avi_fname
    make_labeled_video_custom_annotations(dlc_config, video_fname, new_dlc_df)

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
    clust_df = pd.read_pickle(df_fname)
    # with open(df_fname, 'rb') as f:
        # clust_df = pickle.load(f)
    options = {'min_length':min_track_length, 'num_frames':total_num_frames, 'verbose':0}
    # Loop through tracklets, and make a video for each
    all_bodyparts = np.asarray(clust_df['clust_ind'])

    neuron_ind = 1 # Only for video indices
    options['neuron_ind'] = 1
    for _, row in clust_df.iterrows():
        this_dlc_df = build_dlc_annotation_one_tracklet(row, all_bodyparts, **options)
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
        video_fname = c.datafiles.red_avi_fname
        make_labeled_video_custom_annotations(dlc_config, video_fname, this_dlc_df,
                                              video_suffix=scorer)

        # Clean up by deleting the temporary .h5/.csv/.pickle files
        [os.remove(f) for f in prev_fnames]
        if (num_videos_to_make is not None) and (neuron_ind > num_videos_to_make):
            break

    if verbose >= 1:
        print(f"Finished making videos for {neuron_ind-1} neurons")


##
## Secondary pipeline: create training data
##


def get_indices_full_overlap(clust_df, which_frames):
    def check_frames(vals, which_frames=which_frames):
        return all([f in vals for f in which_frames])

    return clust_df['slice_ind'].apply(check_frames)


def build_relative_imagenames(raw_video_fname,
                              png_fnames=None,
                              num_frames=None,
                              file_ext='tif'):

    if png_fnames is None and num_frames is None:
        print("Error: one of png_fnames or num_frames must be passed")
        raise ValueError
    if png_fnames is None:
        png_fnames = [f'img{i}.{file_ext}' for i in range(num_frames)]

    relative_imagenames = []
    folder_name = os.path.join('labeled-data',os.path.basename(raw_video_fname)[:8])
    for f in png_fnames:
        relative_imagenames.append(os.path.join(folder_name, f))
    return relative_imagenames, folder_name


def build_tif_training_data(c, which_frames, preprocessing_settings=None, verbose=0):

    video_fname = c.datafiles.red_bigtiff_fname
    num_z = c.preprocessing.num_total_slices
    # Get the file names
    dlc_config = auxiliaryfunctions.read_config(c.tracking.DLC_config_fname)
    project_folder = dlc_config['project_path']
    out = build_relative_imagenames(video_fname, num_frames=len(which_frames))
    relative_imagenames, subfolder_name = out

    # Initilize the training data subfolder
    full_subfolder_name = os.path.join(project_folder, subfolder_name)
    if not os.path.isdir(full_subfolder_name):
        os.mkdir(full_subfolder_name)

    # Write the tif files
    print('Writing tif files...')
    for i, rel_fname in tqdm(zip(which_frames, relative_imagenames), total=len(which_frames)):
        dat = c.get_single_volume(i)
        if preprocessing_settings is not None:
            dat = perform_preprocessing(dat, preprocessing_settings)
        fname = os.path.join(project_folder, rel_fname)
        tifffile.imwrite(fname, dat)

    if verbose >= 0:
        print(f"{len(which_frames)} tif files written in project {full_subfolder_name}")

    return relative_imagenames, full_subfolder_name


def OLD_training_data_3d_from_annotations(config,
                                          df_fname,
                                          which_frames,
                                          scorer=None,
                                          total_num_frames=500,
                                          coord_names=['x','y','likelihood'],
                                          preprocessing_settings=None,
                                          verbose=0):
    """
    Creates a set of training frames or volumes starting from a saved dataframe of tracklets

    Takes frames in the list which_frames, taking neurons that are present in each
    """

    c = load_config(config)

    # Load the dataframe name, and produce DLC-style annotations
    clust_df = pd.read_pickle(df_fname)

    # Build a sub-df with only the relevant neurons and slices
    subset_df = build_subset_df_from_tracklets(clust_df, which_frames)

    # Save the individual tif files
    out = build_tif_training_data(c, which_frames, preprocessing_settings=preprocessing_settings)
    relative_imagenames, full_subfolder_name = out

    # Cast the dataframe in DLC format
    options = {'min_length':0, 'num_frames':total_num_frames,
           'coord_names':coord_names,
           'verbose':verbose,
           'relative_imagenames':relative_imagenames,
           'which_frame_subset':which_frames}
    new_dlc_df = build_dlc_annotation_from_tracklets(subset_df, **options)
    if new_dlc_df is None:
        print("Found no tracks long enough; aborting")
        return None

    # Save annotations using DLC-style names, and update the config files
    options = {'project_folder':full_subfolder_name}
    out = save_dlc_annotations(scorer, df_fname, c, new_dlc_df, **options)[0]
    build_dlc_name = out
    synchronize_config_files(c, build_dlc_name,
                             dummy_subfolder=full_subfolder_name,
                             scorer=scorer,
                             num_dims=len(coord_names)-1)

    # Finally,

    # # Finally, make the video
    # dlc_config = c.tracking.DLC_config_fname
    # video_fname = c.datafiles.red_avi_fname
    # make_labeled_video_custom_annotations(dlc_config, video_fname, new_dlc_df)

    return new_dlc_df
