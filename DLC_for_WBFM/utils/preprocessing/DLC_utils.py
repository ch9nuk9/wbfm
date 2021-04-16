import deeplabcut
from DLC_for_WBFM.bin.configuration_definition import load_config, DLCForWBFMTracking, save_config
from DLC_for_WBFM.utils.video_and_data_conversion.video_conversion_utils import write_video_projection_from_ome_file_subset
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords3d
# from DLC_for_WBFM.utils.feature_detection.visualize_using_dlc import training_data_from_annotations
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
    # Actually make projects
    all_avi_fnames = []
    all_dlc_configs = []
    for center in all_center_slices:
        # Make minimax video from btf
        which_slices = get_which_slices(center, num_crop_slices)
        vid_opt['which_slices'] = which_slices
        this_avi_fname = write_video_projection_from_ome_file_subset(**vid_opt)
        # Make dlc project
        dlc_opt['label'] = f"-c{center}"
        dlc_opt['video_path'] = this_avi_fname
        this_dlc_config = create_dlc_project(dlc_opt)
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

    opt = {}
    opt['df_fname'] = df_fname
    opt['scorer'] = scorer
    opt['total_num_frames'] = _config['dataset_params']['num_frames']
    opt['coord_names'] = ['x','y','likelihood']
    opt['which_frames'] = _config['training_data_3d']['which_frames']
    # TODO
    # new_dlc_df = training_data_from_annotations(**opt)
