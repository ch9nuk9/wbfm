from DLC_for_WBFM.utils.feature_detection.visualize_using_dlc import training_data_from_annotations
from DLC_for_WBFM.utils.feature_detection.utils_pipeline import track_neurons_full_video
from DLC_for_WBFM.utils.preprocessing.DLC_utils import create_dlc_project
from DLC_for_WBFM.utils.feature_detection.utils_tif import PreprocessingSettings
from DLC_for_WBFM.utils.video_and_data_conversion.video_conversion_utils import write_video_projection_from_ome_file_subset
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords3d

import os
import os.path as osp
import numpy as np
import pandas as pd
import pickle
import tifffile

###
### For use with produces tracklets (step 2 of pipeline)
###

def partial_track_video_using_config(vid_fname, _config, DEBUG=False):
    """
    Produce training data via partial tracking using 3d feature-based method

    This function is designed to be used with an external .yaml config file

    See new_project_defaults/2-training_data/training_data_config.yaml
    See also track_neurons_full_video()
    """

    # Load preprocessing settings
    p_fname = _config['preprocessing_config']
    p = PreprocessingSettings.load_from_yaml(p_fname)

    ########################
    # Make tracklets
    ########################
    # Get options
    opt = _config['tracker_params']
    opt['num_frames'] = _config['dataset_params']['num_frames']
    if DEBUG:
        opt['num_frames'] = 2
    opt['start_frame'] = _config['dataset_params']['start_volume']
    opt['num_slices'] = _config['dataset_params']['num_slices']

    out = track_neurons_full_video(vid_fname,
                                   preprocessing_settings=p,
                                   **opt)
    ########################
    # Postprocess matches
    ########################
    b_matches, b_conf, b_frames, b_candidates = out
    new_candidates = fix_candidates_without_confidences(b_candidates)
    bp_matches = calc_all_bipartite_matches(new_candidates)
    df = build_tracklets_from_classes(b_frames, bp_matches)

    ########################
    # Save matches to disk
    ########################
    subfolder = osp.join('2-training_data', 'raw')
    os.mkdir(subfolder)

    fname = osp.join(subfolder, 'clust_df_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(df,f)
    fname = osp.join(subfolder, 'match_dat.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(b_matches, f)
    fname = osp.join(subfolder, 'frame_dat.pickle')
    [frame.prep_for_pickle() for frame in b_frames.values()]
    with open(fname, 'wb') as f:
        pickle.dump(b_frames, f)


###
### For use with training a stack of DLC (step 3 of pipeline)
###

def create_dlc_training_from_tracklets(vid_fname, _config,
                                       scorer=None,
                                       task_name=None):

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
    which_frames = best_tracklet_covering(df, **tracklet_opt)
    # Also save these chosen frames
    updates = {'which_frames': which_frames}
    _config['training_data_3d'].update(updates)
    edit_config(_config['self_path'], _config)

    ########################
    # Initialize the DLC projects
    ########################

    num_crop_slices: _config['training_data_2d']['num_crop_slices']
    all_center_slices = _config['training_data_2d']['all_center_slices']

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
    del vid_opt['red_and_green_mirrored']

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
        # Make videos
        which_slices = get_which_slices(center, num_crop_slices)
        vid_opt['which_slices'] = which_slices
        this_avi_fname = write_video_projection_from_ome_file_subset(**vid_opt)
        # Make dlc project
        dlc_opt['label'] = f"-c{center}"
        dlc_opt['video_path'] = this_avi_fname
        this_dlc_config = create_dlc_project(dlc_opt)
        # Save
        all_avi_fnames.append(this_avi_fname)
        all_dlc_configs.append(this_dlc_config)

    # Then delete the created avis... they are copied into the DLC folder
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
    new_dlc_df = training_data_from_annotations(**opt)
