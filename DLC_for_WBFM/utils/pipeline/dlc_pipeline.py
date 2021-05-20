from DLC_for_WBFM.utils.preprocessing.convert_matlab_annotations_to_DLC import csv_annotations2config_names
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings, perform_preprocessing
from DLC_for_WBFM.utils.video_and_data_conversion.video_conversion_utils import write_numpy_as_avi
from DLC_for_WBFM.utils.projects.utils_project import edit_config
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import best_tracklet_covering
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.preprocessing.DLC_utils import get_annotations_from_dlc_config, get_z_from_dlc_name, update_pose_config, training_data_from_annotations, \
    create_dlc_project
import pandas as pd
import numpy as np
import tifffile
import os
from tqdm import tqdm
import deeplabcut


###
### For use with training a stack of DLC (step 3 of pipeline)
###

def create_dlc_training_from_tracklets(vid_fname,
                                       config,
                                       scorer=None,
                                       task_name=None,
                                       verbose=0,
                                       DEBUG=False):

    df_fname = config['training_data_3d']['annotation_fname']
    df = pd.read_pickle(df_fname)

    all_center_slices, which_frames = _get_frames_for_dlc_training(DEBUG, config, df)

    all_avi_fnames, preprocessed_dat, vid_opt, video_exists = _prep_videos_for_dlc(DEBUG, all_center_slices, config,
                                                                                   verbose, vid_fname, which_frames)

    dlc_opt, net_opt, png_opt = _define_project_options(config, df, scorer, task_name)
    # Actually make projects
    all_dlc_configs = []
    for i, center in enumerate(all_center_slices):
        _initialize_project_from_btf(all_avi_fnames, all_dlc_configs, center, dlc_opt, i, net_opt, png_opt,
                                     preprocessed_dat, vid_opt, video_exists)

    # Then delete the created avis because they are copied into the DLC folder
    # [os.remove(f) for f in all_avi_fnames]

    # Save list of dlc config names
    config['dlc_projects']['all_configs'] = all_dlc_configs
    edit_config(config['self_path'], config)


def _prep_videos_for_dlc(DEBUG, all_center_slices, config, verbose, vid_fname, which_frames):
    # OPTIMIZE: for now, requires re-preprocessing
    video_exists = []
    all_avi_fnames = []
    for center in all_center_slices:
        # Make minimax video from btf
        this_avi_fname = _make_avi_name(center)
        all_avi_fnames.append(this_avi_fname)
        if os.path.exists(this_avi_fname):
            print(f"Using video at: {this_avi_fname}")
            video_exists.append(True)
        else:
            video_exists.append(False)
    # IF videos are required, then prep the data
    if all(video_exists):
        print("All required videos exist; no preprocessing necessary")
        preprocessed_dat = []
        _, vid_opt = _get_video_options(config, vid_fname)
    else:
        preprocessed_dat, vid_opt = _preprocess_all_frames(DEBUG, config, verbose, vid_fname, which_frames)
    return all_avi_fnames, preprocessed_dat, vid_opt, video_exists


def _define_project_options(config, df, scorer, task_name):
    # Get dlc project and naming options
    dlc_opt = {'task_name': task_name,
               'experimenter': scorer,
               'working_directory': '3-tracking',
               'copy_videos': True}
    # Get a few frames as training data
    png_opt = {'df_fname': df, 'total_num_frames': config['dataset_params']['num_frames'], 'coord_names': ['x', 'y'],
               'which_frames': config['training_data_3d']['which_frames'],
               'max_z_dist_for_traces': config['training_data_2d']['max_z_dist_for_traces']}
    # png_opt['scorer'] = scorer
    # Connecting these frames to a network architecture
    net_opt = {'net_type': "resnet_50",  # 'mobilenet_v2_0.35' #'resnet_50
               'augmenter_type': "imgaug"}
    return dlc_opt, net_opt, png_opt


def _initialize_project_from_btf(all_avi_fnames, all_dlc_configs, center, dlc_opt, i, net_opt, png_opt,
                                 preprocessed_dat, vid_opt, video_exists):
    # Make or get video
    this_avi_fname = all_avi_fnames[i]
    if not video_exists[i]:
        vid_opt['out_fname'] = this_avi_fname
        write_numpy_as_avi(preprocessed_dat[:, center, ...], **vid_opt)
    # Make dlc project
    dlc_opt['label'] = f"-c{center}"
    dlc_opt['video_path'] = this_avi_fname
    this_dlc_config = create_dlc_project(**dlc_opt)
    # Training frame extraction
    png_opt['which_z'] = center
    png_opt['dlc_config_fname'] = this_dlc_config
    png_opt['vid_fname'] = this_avi_fname
    ann_fname = training_data_from_annotations(**png_opt)[1]
    # Syncronize the dlc_config with the annotations
    csv_annotations2config_names(this_dlc_config, ann_fname, num_dims=2)
    # Format the training data
    deeplabcut.create_training_dataset(this_dlc_config, **net_opt)
    update_pose_config(this_dlc_config)
    # Save to list
    all_dlc_configs.append(this_dlc_config)


def _get_frames_for_dlc_training(DEBUG, config, df):
    # Choose a subset of frames with enough tracklets
    num_frames_needed = config['training_data_3d']['num_training_frames']
    tracklet_opt = {'num_frames_needed': num_frames_needed,
                    'num_frames': config['dataset_params']['num_frames'],
                    'verbose': 1}
    if DEBUG:
        tracklet_opt['num_frames_needed'] = 2
    which_frames, _ = best_tracklet_covering(df, **tracklet_opt)
    # Also save these chosen frames
    updates = {'which_frames': which_frames}
    config['training_data_3d'].update(updates)
    edit_config(config['self_path'], config)
    all_center_slices = config['training_data_2d']['all_center_slices']
    if DEBUG:
        all_center_slices = [all_center_slices[0]]
    return all_center_slices, which_frames


def _make_avi_name(center):
    fname = f"center{center}.avi"  # NOT >8 CHAR (without .avi)
    if len(fname) > 12:
        # BUG: fix required short filenames
        # Another function clips labeled-data/folder-name at 8 chars
        # But, that name must be the same as the video
        raise ValueError(f"Bug if this is too long {fname}")
    return fname


def _preprocess_all_frames(DEBUG, config, verbose, vid_fname, which_frames):
    sz, vid_opt = _get_video_options(config, vid_fname)
    if verbose >= 1:
        print("Preprocessing data, this could take a while...")
    p = PreprocessingSettings.load_from_yaml(config['preprocessing_config'])
    start_volume = config['dataset_params']['start_volume']
    num_total_frames = start_volume + config['dataset_params']['num_frames']
    num_slices = config['dataset_params']['num_slices']
    if DEBUG:
        # Make a much shorter video
        num_total_frames = which_frames[-1] + 1
    preprocessed_dat = np.zeros((num_total_frames, num_slices) + sz, dtype='uint16')
    # Load data and preprocess
    frame_list = list(range(num_total_frames))
    for i in tqdm(frame_list):
        preprocessed_dat[i, ...] = _get_and_preprocess(i, num_slices, p, start_volume, vid_fname)
    return preprocessed_dat, vid_opt


def _get_video_options(config, vid_fname):
    with tifffile.TiffFile(vid_fname) as tif:
        sz = tif.pages[0].shape
    vid_opt = {'fps': config['dataset_params']['fps'],
               'frame_height': sz[0],
               'frame_width': sz[1]}
    return sz, vid_opt


def _get_and_preprocess(i, num_slices, p, start_volume, vid_fname):
    dat_raw = get_single_volume(vid_fname, i, num_slices, dtype='uint16')
    # Don't preprocess data that we didn't even segment!
    if i >= start_volume:
        # preprocessed_dat[i, ...] = perform_preprocessing(dat_raw, p)
        return perform_preprocessing(dat_raw, p)
    else:
        # preprocessed_dat[i, ...] = dat_raw
        return dat_raw


def train_all_dlc_from_config(config):
    """
    Simple multi-network wrapper around:
    deeplabcut.train_network()
    """
    from tensorflow.errors import CancelledError
    all_dlc_configs = config['dlc_projects']['all_configs']

    print(f"Found {len(all_dlc_configs)} networks; beginning training")
    for dlc_config in all_dlc_configs:
        # Check to see if already trained
        try:
            deeplabcut.evaluate_network(dlc_config)
            print(f"Network for config {dlc_config} already trained; skipping")
            continue
        except FileNotFoundError:
            # Not yet trained, so train it!
            pass
        try:
            deeplabcut.train_network(dlc_config)
        except CancelledError:
            # This means it finished the planned number of steps
            pass


def make_3d_tracks_from_stack(track_cfg, DEBUG=False):
    """
    Applies trained DLC networks to full video and collects into 3d track
    """

    all_dlc_configs = track_cfg['dlc_projects']['all_configs']

    # Apply networks
    all_dfs = []
    neuron2z_dict = {}
    i_neuron = 0
    for dlc_config in all_dlc_configs:
        _analyze_video_and_save_tracks(DEBUG, all_dfs, dlc_config, i_neuron, neuron2z_dict)

    final_df = pd.concat(all_dfs, axis=1)
    # Collect 2d data
    # i.e. just add the z coordinate to it
    # For some reason, the concat after adding z was broken :(
    for name, z in neuron2z_dict.items():
        final_df[name, 'z'] = z
    final_df.sort_values('bodyparts', axis=1, inplace=True)
    if DEBUG:
        print(final_df)

    # Save dataframe
    dest_folder = '3-tracking'
    fname = os.path.join(dest_folder, 'full_3d_tracks.h5')
    final_df.to_hdf(fname, "df_with_missing")

    # Save in yaml
    udpates = {'final_3d_tracks': {'df_fname': fname}}
    edit_config(track_cfg['self_path'], udpates)

    return final_df


def _analyze_video_and_save_tracks(DEBUG, all_dfs, dlc_config, i_neuron, neuron2z_dict):
    dlc_cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_config)
    video_list = list(dlc_cfg['video_sets'].keys())
    # Works even if already analyzed; skips if empty
    try:
        deeplabcut.analyze_videos(dlc_config, video_list)
    except IndexError:
        # Doesn't append anything to all_dfs
        print(f"No neurons found; skipping project {dlc_config}")
        return
    # Get data for later use
    df_fname = get_annotations_from_dlc_config(dlc_config)
    if DEBUG:
        print(f"Using 2d annotations: {df_fname}")
    # Remove scorer and rename neurons
    df = pd.read_hdf(df_fname)
    df_scorer = df.columns.values[0][0]
    df = df[df_scorer]
    i_neuron_new = i_neuron + len(df.columns.levels[0])
    neuron_range = range(i_neuron, i_neuron_new)
    i_neuron = i_neuron_new
    new_names = [f'neuron{i}' for i in neuron_range]
    z = get_z_from_dlc_name(dlc_config)
    neuron2z_dict.update({n: z for n in new_names})
    df.columns.set_levels(new_names, level=0, inplace=True)
    all_dfs.append(df)


