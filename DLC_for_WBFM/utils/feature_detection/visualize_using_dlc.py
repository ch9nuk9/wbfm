import deeplabcut
from deeplabcut.utils.make_labeled_video import CreateVideo
from deeplabcut.utils.video_processor import VideoProcessorCV as vp
from deeplabcut.utils import auxiliaryfunctions
from DLC_for_WBFM.bin.configuration_definition import load_config, DLCForWBFMTracking, save_config
import os
import tifffile
from DLC_for_WBFM.utils.preprocessing.convert_matlab_annotations_to_DLC import csv_annotations2config_names
# from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
import numpy as np
import pandas as pd
from tqdm import tqdm

##
## Helper functions for converting annotations
##

def build_dlc_annotation_one_tracklet(row,
                                      bodypart,
                                      num_frames=1000,
                                      coord_names=None,
                                      which_frame_subset=None,
                                      min_length=5,
                                      neuron_ind=1,
                                      relative_imagenames=None,
                                      verbose=0):
    """
    Builds DLC-style dataframe and .h5 annotation from my tracklet dataframe

    Can also be 3d if coord_names is passed as ['z', 'x', 'y', 'likelihood']
    """
    if coord_names is None:
        coord_names = ['x','y','likelihood']
    # TODO: Check z

    # Variables to be written
    scorer = 'feature_tracker'
    if relative_imagenames is None:
        # Just frame number
        index = list(range(num_frames))
    else:
        index = relative_imagenames

    tracklet_length = len(row['all_xyz'])

    if verbose >= 2:
        print(f"Found tracklet of length {tracklet_length}")
    if tracklet_length < min_length:
        return None

    # Relies on ZXY format for this_xyz column in the original dataframe
    coord_mapping = {'z':0, 'x':1, 'y':2}

    # Build a dataframe for one neuron across all frames
    # Will be zeros if not detected in a given frame
    coords = np.zeros((num_frames,len(coord_names),))
    for this_slice, this_xyz, this_prob in zip(row['slice_ind'], row['all_xyz'], row['all_prob']):
        # TODO: only works for xy; this_xyz is format ZXY
        for i, coord_name in enumerate(coord_names):
            if coord_name in coord_mapping:
                # is spatial
                coords[this_slice,i] = int(this_xyz[coord_mapping[coord_name]])
            else:
                # is non-spatial, i.e. likelihood
                try:
                    coords[this_slice,-1] = this_prob
                except:
                    coords[this_slice,-1] = 0.0
                    pass
    if which_frame_subset is not None:
        # error
        coords = coords[which_frame_subset,:]

    m_index = pd.MultiIndex.from_product([[scorer], [bodypart],
                                        coord_names],
                                        names=['scorer', 'bodyparts', 'coords'])
    frame = pd.DataFrame(coords, columns = m_index, index = index)

    return frame


def build_dlc_annotation_all(clust_df, min_length, num_frames=1000,
                             coord_names=['x','y','likelihood'],
                             relative_imagenames=None,
                             which_frame_subset=None,
                             verbose=1):
    new_dlc_df = None
    # all_bodyparts = np.asarray(clust_df['clust_ind'])

    neuron_ind = 1
    opt = {'min_length':min_length, 'verbose':verbose-1,
           'num_frames':num_frames,
           'coord_names':coord_names,
           'relative_imagenames':relative_imagenames,
           'which_frame_subset':which_frame_subset}
    for i, row in tqdm(clust_df.iterrows(), total=clust_df.shape[0]):
        opt['neuron_ind'] = neuron_ind
        ind = row['clust_ind']
        bodypart = f'neuron{ind}'
        frame = build_dlc_annotation_one_tracklet(row, bodypart, **opt)
        if frame is not None:
            new_dlc_df = pd.concat([new_dlc_df, frame],axis=1)
            neuron_ind = neuron_ind + 1
#         if verbose >= 1:
#             print("============================")
#             print(f"Row {i}/{len(all_bodyparts)}")
    if verbose >= 1 and new_dlc_df is not None:
        print(f"Found {len(new_dlc_df.columns)/len(coord_names)} tracks of length >{min_length}")

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


def synchronize_config_files(c, build_dlc_name, num_dims=2):
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
    csv_annotations2config_names(dlc_config, csv_annotations, num_dims=num_dims)
    

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
    opt = {'min_length':min_track_length, 'num_frames':total_num_frames,
           'coord_names':coord_names,
           'verbose':verbose}
    new_dlc_df = build_dlc_annotation_all(clust_df, **opt)
    if new_dlc_df is None:
        print("Found no tracks long enough; aborting")
        return None

    # Save annotations using DLC-style names, and update the config files
    build_dlc_name = save_dlc_annotations(scorer, df_fname, c, new_dlc_df)[0]
    synchronize_config_files(c, build_dlc_name, num_dims=len(coord_names)-1)

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
    clust_df = pd.read_pickle(df_fname)
    # with open(df_fname, 'rb') as f:
        # clust_df = pickle.load(f)
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


##
## Secondary pipeline: create training data
##


def get_indices_full_overlap(clust_df, which_frames):
    def check_frames(vals, which_frames=which_frames):
        return all([f in vals for f in which_frames])

    return clust_df['slice_ind'].apply(check_frames)


def build_subset_df(clust_df, which_frames):

    def check_frames(test_frames, which_frames=which_frames):
        test_frames_set = set(test_frames)
        local2global_ind = {}
        for f in which_frames:
            if f in test_frames_set:
                local2global_ind[test_frames.index(f)] = f
            else:
                # Must all be present
                return None
        return local2global_ind

    def keep_subset(this_ind_dict, old_ind):
        new_ind = []
        for i in this_ind_dict:
            try:
                new_ind.append(old_ind[i])
            except:
                continue
        return new_ind

    def rename_slices(this_ind_dict):
        return list(this_ind_dict.keys())
        # return which_frames

    sub_df = clust_df.copy()
    # Get only the covering neurons
    which_neurons = sub_df['slice_ind'].apply(check_frames)
    which_neurons_dict = which_neurons.to_dict()
    to_keep = [(v is not None) for k,v in which_neurons_dict.items()]
    for i, val in enumerate(to_keep):
        if not val:
            del which_neurons_dict[i]
    which_neurons_df = sub_df[to_keep]

    # Get only the indices of those neurons corresponding to these frames
    names = {'all_xyz':'all_xyz_old',
            'all_ind_local':'all_ind_local_old',
            'all_prob':'all_prob_old',
            'slice_ind':'slice_ind_old'}
    out_df = which_neurons_df.rename(columns=names)

    # All 4 fields that were renamed
    f0 = lambda df : keep_subset(which_neurons_dict[df['clust_ind']], df['all_ind_local_old'])
    out_df['all_ind_local'] = out_df.apply(f0, axis=1)

    f1 = lambda df : keep_subset(which_neurons_dict[df['clust_ind']], df['all_xyz_old'])
    out_df['all_xyz'] = out_df.apply(f1, axis=1)

    f2 = lambda df : keep_subset(which_neurons_dict[df['clust_ind']], df['all_prob_old'])
    out_df['all_prob'] = out_df.apply(f2, axis=1)

    # Final one is slightly different
    # f3 = lambda df : rename_slices(which_neurons_dict[df['clust_ind']])
    f3 = lambda df : which_frames
    out_df['slice_ind'] = out_df.apply(f3, axis=1)

    return out_df


def build_relative_imagenames(c, png_fnames=None, num_frames=None):

    if png_fnames is None and num_frames is None:
        print("Error: one of png_fnames or num_frames must be passed")
        raise ValueError
    if png_fnames is None:
        png_fnames = [f'img{i}.tif' for i in range(num_frames)]
    raw_video_fname = c.datafiles.red_bigtiff_fname

    relative_imagenames = []
    folder_name = os.path.join('labeled-data',os.path.basename(raw_video_fname)[:8])
    for f in png_fnames:
        relative_imagenames.append(os.path.join(folder_name, f))
    return relative_imagenames, folder_name


def build_tif_training_data(c, which_frames, verbose=0):

    # Get the file names
    dlc_config = auxiliaryfunctions.read_config(c.tracking.DLC_config_fname)
    project_folder = dlc_config['project_path']
    out = build_relative_imagenames(c, num_frames=len(which_frames))
    relative_imagenames, subfolder_name = out

    video_fname = c.datafiles.red_bigtiff_fname
    num_z = c.preprocessing.num_total_slices

    # Initilize the training data subfolder
    full_subfolder_name = os.path.join(project_folder, subfolder_name)
    if not os.path.isdir(full_subfolder_name):
        os.mkdir(full_subfolder_name)

    # Write the tif files
    print('Writing tif files...')
    for i, rel_fname in tqdm(zip(which_frames, relative_imagenames), total=len(which_frames)):
        dat = c.get_single_volume(i)
        fname = os.path.join(project_folder, rel_fname)
        tifffile.imwrite(fname, dat)

    if verbose >= 0:
        print(f"{len(which_frames)} tif files written in project {full_subfolder_name}")

    return relative_imagenames


def training_data_from_annotations(config, df_fname,
                                   which_frames,
                                   scorer=None,
                                   total_num_frames=500,
                                   coord_names=['x','y','likelihood'],
                                   verbose=0):
    """
    Creates a set of training frames or volumes starting from a saved dataframe of tracklets

    Takes frames in the list which_frames, taking neurons that are present in each
    """

    c = load_config(config)

    # Load the dataframe name, and produce DLC-style annotations
    # with open(df_fname, 'rb') as f:
    #     clust_df = pickle.load(f)
    clust_df = pd.read_pickle(df_fname)

    # Build a sub-df with only the relevant neurons and slices
    subset_df = build_subset_df(clust_df, which_frames)

    # Save the individual tif files
    relative_imagenames = build_tif_training_data(c, which_frames)
    # out = build_relative_imagenames(c, num_frames=len(which_frames))
    # relative_imagenames, tif_fnames = out

    # Cast the dataframe in DLC format
    opt = {'min_length':0, 'num_frames':total_num_frames,
           'coord_names':coord_names,
           'verbose':verbose,
           'relative_imagenames':relative_imagenames,
           'which_frame_subset':which_frames}
    new_dlc_df = build_dlc_annotation_all(subset_df, **opt)
    if new_dlc_df is None:
        print("Found no tracks long enough; aborting")
        return None

    # Save annotations using DLC-style names, and update the config files
    build_dlc_name = save_dlc_annotations(scorer, df_fname, c, new_dlc_df)[0]
    synchronize_config_files(c, build_dlc_name, num_dims=len(coord_names)-1)

    # Finally,

    # # Finally, make the video
    # dlc_config = c.tracking.DLC_config_fname
    # vid_fname = c.datafiles.red_avi_fname
    # make_labeled_video_custom_annotations(dlc_config, vid_fname, new_dlc_df)

    return new_dlc_df
