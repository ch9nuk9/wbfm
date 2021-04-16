import deeplabcut
from DLC_for_WBFM.bin.configuration_definition import load_config, DLCForWBFMTracking, save_config
from DLC_for_WBFM.utils.video_and_data_conversion.video_conversion_utils import write_video_projection_from_ome_file_subset
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords3d
from DLC_for_WBFM.utils.feature_detection.visualize_using_dlc import build_subset_df, build_tif_training_data, build_dlc_annotation_all
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config
import pandas as pd
import numpy as np
import tifffile


##
## Functions for building DLC projects using config class
##

def create_dlc_project_from_config(config, label='',copy_videos=True):
    """
    Creates a DLC subproject within a parent folder defined by config

    Note: copy_videos is required on Windows
    """

    c = load_config(config)

    # Force shorter name
    dlc_opt = {'project':c.task_name[0] + label,
               'experimenter':c.experimenter[0],
               'videos':[c.datafiles.red_avi_fname],
               'copy_videos':copy_videos,
               'working_directory':c.get_dirname()}

    dlc_config_fname = deeplabcut.create_new_project(**dlc_opt)

    tracking = DLCForWBFMTracking(dlc_config_fname)
    c.tracking = tracking
    save_config(c)

    return dlc_config_fname


def create_dlc_project(task_name,
                       experimenter,
                       video_path,
                       working_directory,
                       label='',
                       copy_videos=True):
    """
    Creates a DLC subproject within working_directory
        Returns the string for the created config file

    Note: copy_videos is required on Windows

    Same function but separate interface as: create_dlc_project_from_config
        i.e. expanded interace
    """

    c = load_config(config)

    # Force shorter name
    dlc_opt = {'project':task_name[0] + label,
               'experimenter':experimenter[0],
               'videos':[video_path],
               'copy_videos':copy_videos,
               'working_directory':working_directory}

    dlc_config_fname = deeplabcut.create_new_project(**dlc_opt)

    return dlc_config_fname


def training_data_from_annotations(vid_fname,
                                   df_fname,
                                   which_frames,
                                   which_z,
                                   dlc_config_fname,
                                   max_z_dist_for_traces=2,
                                   scorer=None,
                                   total_num_frames=500,
                                   coord_names=None,
                                   preprocessing_settings=None,
                                   verbose=0):
    """
    Creates a set of training frames or volumes starting from a saved dataframe of tracklets
    Takes frames in the list which_frames, taking neurons that are present in each

    Parameters
    =================
    vid_fname: str
        Path of the 2d avi video corresponding, from which frames will be taken
    df_fname: str or pd.DataFrame
        Path or object (DataFrame) with 3d annotations
        Frame indices correspond to original btf
    which_frames: list
        List of indices that will be take from vid_fname as training frames
    which_z: int
        Which z-slice corresponding to this video
    dlc_config_fname: str
        The location of the dlc config.yaml file
        The pngs will be written in a subfolder of this parent
    max_z_dist_for_traces: int
        How many slices away a neuron centroid can be from which_z
        i.e. dist <= which_z +- max_z_dist_for_traces
    scorer: str
        Name that will be written in the DLC dataframes
    total_num_frames: int
        Total number of frames in the video
    coord_names: list of str
        Which coordinates to save, in order. Options are:
        ['x', 'y', 'z', 'likelihood']
    preprocessing_settings: None or PreprocessingSettings object
        If the training frames need to be preprocessed after being taken from the .avi video
        Default is None
    verbose: int
        Amount to print

    See also: training_data_3d_from_annotations
    """

    if coord_names is None:
        coord_names = ['x', 'y', 'likelihood']

    # Load the dataframe name, and produce DLC-style annotations
    if type(df_fname)==str:
        clust_df = pd.read_pickle(df_fname)
    else:
        assert type(df_fname)==pd.DataFrame, "Must pass dataframe or filename of dataframe"
        clust_df = df_fname

    # Build a sub-df with only the relevant neurons and slices
    # TODO: z frame constraint
    subset_df = build_subset_df(clust_df, which_frames)

    # TODO: Save the individual png files
    # out = build_tif_training_data(c, which_frames, preprocessing_settings=preprocessing_settings)
    relative_imagenames, full_subfolder_name = out

    # Cast the dataframe in DLC format
    opt = {'min_length':0,
           'num_frames':total_num_frames,
           'coord_names':coord_names,
           'verbose':verbose,
           'relative_imagenames':relative_imagenames,
           'which_frame_subset':which_frames}
    new_dlc_df = build_dlc_annotation_all(subset_df, **opt)
    if new_dlc_df is None:
        print("Found no tracks long enough; aborting")
        return None

    # Save annotations using DLC-style names, and update the config files
    opt = {'project_folder':full_subfolder_name}
    out = save_dlc_annotations(scorer, df_fname, c, new_dlc_df, **opt)[0]
    build_dlc_name = out
    # synchronize_config_files(c, build_dlc_name,
    #                          dummy_subfolder=full_subfolder_name,
    #                          scorer=scorer,
    #                          num_dims=len(coord_names)-1)

    return new_dlc_df


###
### For use with training a stack of DLC (step 3 of pipeline)
###

def create_dlc_training_from_tracklets(vid_fname, _config,
                                       scorer=None,
                                       task_name=None,
                                       DEBUG=False):

    ########################
    # Load annotations
    ########################
    df_fname = _config['3d_training_data']['annotation_fname']
    df = pd.load_from_pickle(df_fname)

    ########################
    # Prepare for dlc-style training data
    ########################

    # Choose a subset of frames with enough tracklets
    num_frames_needed = _config['training_data_3d']['num_training_frames']
    tracklet_opt = {'num_frames_needed': num_frames_needed,
                    'num_frames': _config['dataset_params']['num_frames'],
                    'verbose':1}
    if DEBUG:
        tracklet_opt['num_frames_needed'] = 2
    which_frames = best_tracklet_covering(df, **tracklet_opt)
    # Also save these chosen frames
    updates = {'which_frames': which_frames}
    _config['training_data_3d'].update(updates)
    edit_config(_config['self_path'], _config)

    ########################
    # Initialize the DLC projects
    ########################

    num_crop_slices = _config['training_data_2d']['num_crop_slices']
    all_center_slices = _config['training_data_2d']['all_center_slices']
    if DEBUG:
        all_center_slices = [all_center_slices[0]]

    # First create the videos, Then, create the project structure
    # TODO: optimize this

    # Get the video options
    with tifffile.TiffFile(vid_fname) as tif:
        frame_height, frame_width = tif.pages[0].shape
    vid_opt = {'frame_height': frame_height,
               'frame_width': frame_width,
               'out_dtype': 'uint16',
               'flip_x': False,
               'video_fname': vid_fname}
    vid_opt.update(_config['dataset_params'])
    del vid_opt['red_and_green_mirrored'] # Extra unneeded parameter

    def get_which_slices(center_slice, num_crop_slices):
        return list( get_crop_coords3d((0,0,center_slice),
                                (1,1,num_crop_slices) )[-1] )
    # Get dlc project and naming options
    dlc_opt = {'task_name': task_name,
               'experimenter': scorer,
               'working_directory': '3-tracking',
               'copy_videos':True}
    # Get a few frames as training data
    png_opt = {}
    png_opt['df_fname'] = df
    png_opt['scorer'] = scorer
    png_opt['total_num_frames'] = _config['dataset_params']['num_frames']
    png_opt['coord_names'] = ['x','y','likelihood']
    png_opt['which_frames'] = _config['training_data_3d']['which_frames']
    png_opt['max_z_dist_for_traces'] = _config['training_data_2d']['max_z_dist_for_traces']
    # Actually make projects
    all_avi_fnames = []
    all_dlc_configs = []
    for center in all_center_slices:
        # Make minimax video from btf
        which_z_slices = get_which_slices(center, num_crop_slices)
        vid_opt['which_slices'] = which_z_slices
        this_avi_fname = write_video_projection_from_ome_file_subset(**vid_opt)
        # Make dlc project
        dlc_opt['label'] = f"-c{center}"
        dlc_opt['video_path'] = this_avi_fname
        this_dlc_config = create_dlc_project(dlc_opt)
        # Training frames
        png_opt['which_z'] = center
        png_opt['dlc_config_fname'] = this_dlc_config
        training_data_from_annotations(**png_opt)
        # Save to list
        all_avi_fnames.append(this_avi_fname)
        all_dlc_configs.append(this_dlc_config)

    # Then delete the created avis because they are copied into the DLC folder
    # TODO

    # Save list of dlc config names
    _config['dlc_projects']['all_configs'] = all_dlc_configs
    edit_config(_config['self_path'], _config)

    ########################
    # Actually produce the training data
    ########################
