import concurrent.futures
import threading
from pathlib import Path

import zarr

from DLC_for_WBFM.utils.preprocessing.convert_matlab_annotations_to_DLC import csv_annotations2config_names
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings, perform_preprocessing
from DLC_for_WBFM.utils.video_and_data_conversion.video_conversion_utils import write_numpy_as_avi
from DLC_for_WBFM.utils.projects.utils_project import edit_config
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import best_tracklet_covering
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.preprocessing.DLC_utils import get_annotations_from_dlc_config, get_z_from_dlc_name, \
    update_pose_config, training_data_from_annotations, \
    create_dlc_project, get_annotations_matching_video_in_folder
import pandas as pd
import numpy as np
import tifffile
import os
from tqdm import tqdm
import deeplabcut


###
### For use with training a stack of DLC (step 3 of pipeline)
###

def create_only_videos(vid_fname, config, verbose=1, DEBUG=False):
    """
    Shortened version of create_dlc_training_from_tracklets() that only creates the videos

    Does not require that training data is present; intended to be used when reusing other networks
    """

    all_center_slices = config['training_data_2d']['all_center_slices']
    which_frames = None
    all_avi_fnames, preprocessed_dat, vid_opt, video_exists = _prep_videos_for_dlc(DEBUG, all_center_slices, config,
                                                                                   verbose, vid_fname, which_frames)

    def parallel_func(i_center):
        i, center = i_center
        _get_or_make_avi(all_avi_fnames, center, i, preprocessed_dat, vid_opt, video_exists)

    for i_center in tqdm(enumerate(all_center_slices)):
        parallel_func(i_center)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_center_slices)) as executor:
    #     # futures = executor.map(parallel_func, enumerate(all_center_slices))
    #     # all_avi_fnames = [f.result() for f in futures]
    #     result_futures = list(map(lambda x: executor.submit(parallel_func, x), enumerate(all_center_slices)))
    #     all_avi_fnames = [f.result() for f in concurrent.futures.as_completed(result_futures)]

    return all_avi_fnames


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
    # all_dlc_configs = []
    # for i, center in enumerate(all_center_slices):
    #     this_dlc_config = _initialize_project_from_btf(all_avi_fnames, center, dlc_opt, i, net_opt, png_opt,
    #                                  preprocessed_dat, vid_opt, video_exists, config)
    #     all_dlc_configs.append(this_dlc_config)

    def parallel_func(i, center):
        _initialize_project_from_btf(all_avi_fnames, center, dlc_opt, i, net_opt, png_opt,
                                     preprocessed_dat, vid_opt, video_exists, config)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_center_slices)) as executor:
        futures = executor.map(parallel_func, enumerate(all_center_slices))
        all_dlc_configs = [f.result() for f in futures]

    # Then delete the created avis because they are copied into the DLC folder
    # [os.remove(f) for f in all_avi_fnames]

    # Save list of dlc config names
    config['dlc_projects']['all_configs'] = all_dlc_configs
    edit_config(config['self_path'], config)


def _prep_videos_for_dlc(DEBUG, all_center_slices, config, verbose, vid_fname, which_frames=None):
    all_avi_fnames, video_exists = _get_and_check_avi_filename(all_center_slices)
    # IF videos are required, then prep the data
    if all(video_exists):
        print("All required videos exist; no preprocessing necessary")
        preprocessed_dat = []
        _, vid_opt = _get_video_options(config, vid_fname)
    else:
        preprocessed_dat, vid_opt = _preprocess_all_frames(DEBUG, config, verbose, vid_fname, which_frames)
    return all_avi_fnames, preprocessed_dat, vid_opt, video_exists


def _get_and_check_avi_filename(all_center_slices, subfolder="3-tracking"):
    # OPTIMIZE: for now, requires re-preprocessing
    video_exists = []
    all_avi_fnames = []
    print(f"Making videos for all centers: {all_center_slices}")
    for center in all_center_slices:
        # Make minimax video from btf
        this_avi_fname = os.path.join(subfolder, _make_avi_name(center))
        all_avi_fnames.append(this_avi_fname)
        if os.path.exists(this_avi_fname):
            print(f"Using video at: {this_avi_fname}")
            video_exists.append(True)
        else:
            video_exists.append(False)
    return all_avi_fnames, video_exists


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


def _initialize_project_from_btf(all_avi_fnames, center, dlc_opt, i, net_opt, png_opt,
                                 preprocessed_dat, vid_opt, video_exists, project_config):
    this_avi_fname = _get_or_make_avi(all_avi_fnames, center, i, preprocessed_dat, vid_opt, video_exists)
    # Make dlc project
    dlc_opt['label'] = f"-c{center}"
    dlc_opt['video_path'] = this_avi_fname
    this_dlc_config = create_dlc_project(**dlc_opt)
    # Training frame extraction
    png_opt['which_z'] = center
    png_opt['dlc_config_fname'] = this_dlc_config
    png_opt['vid_fname'] = this_avi_fname
    ann_fname = training_data_from_annotations(**png_opt)[1]
    if ann_fname is not None:
        # Synchronize the dlc_config with the annotations
        csv_annotations2config_names(this_dlc_config, ann_fname, num_dims=2, to_add_skeleton=True)
        # Format the training data
        deeplabcut.create_training_dataset(this_dlc_config, **net_opt)
        update_pose_config(this_dlc_config, project_config)
        # Save to list
        return this_dlc_config
    else:
        return None


def _get_or_make_avi(all_avi_fnames, center, i, preprocessed_dat, vid_opt, video_exists):
    # Make or get video
    out_folder = "3-tracking"
    this_avi_fname = os.path.join(out_folder, all_avi_fnames[i])
    if not video_exists[i]:
        vid_opt['out_fname'] = this_avi_fname
        write_numpy_as_avi(preprocessed_dat[:, center, ...], **vid_opt)
    return this_avi_fname


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


def _preprocess_all_frames(DEBUG, config, verbose, vid_fname, which_frames=None):
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
    chunk_sz = (num_slices, ) + sz
    total_sz = (num_total_frames, ) + chunk_sz
    # preprocessed_dat = np.zeros(total_sz, dtype='uint16'
    preprocessed_dat = zarr.zeros(total_sz, chunks=chunk_sz, dtype='uint16',
                                  synchronizer=zarr.ThreadSynchronizer())
    # read_lock = threading.Lock()
    # Load data and preprocess
    frame_list = list(range(num_total_frames))
    with tifffile.TiffFile(vid_fname) as vid_stream:
        # def parallel_func(i):
        #     preprocessed_dat[i, ...] = _get_and_preprocess(i, num_slices, p, start_volume, vid_stream, read_lock)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=len(frame_list)) as executor:
        #     futures = executor.map(parallel_func, frame_list)
        #     [f.result() for f in futures]
        for i in tqdm(frame_list):
            preprocessed_dat[i, ...] = _get_and_preprocess(i, num_slices, p, start_volume, vid_stream)
    return preprocessed_dat, vid_opt


def _get_video_options(config, vid_fname):
    with tifffile.TiffFile(vid_fname) as tif:
        sz = tif.pages[0].shape
    vid_opt = {'fps': config['dataset_params']['fps'],
               'frame_height': sz[0],
               'frame_width': sz[1]}
    return sz, vid_opt


def _get_and_preprocess(i, num_slices, p, start_volume, vid_fname, read_lock=None):
    if read_lock is None:
        dat_raw = get_single_volume(vid_fname, i, num_slices, dtype='uint16')
    else:
        with read_lock:
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


def make_3d_tracks_from_stack(track_cfg, use_dlc_project_videos=True, DEBUG=False):
    """
    Applies trained DLC networks to full 2d videos and collects into 3d track

    Can be used with the videos from the DLC projects, or external ones
    """

    all_dlc_configs = track_cfg['dlc_projects']['all_configs']

    # Apply networks
    all_dfs = []
    neuron2z_dict = {}
    i_neuron = 0
    if use_dlc_project_videos:
        external_videos = [None for _ in all_dlc_configs]
    else:
        all_center_slices = track_cfg['training_data_2d']['all_center_slices']
        external_videos, videos_exist = _get_and_check_avi_filename(all_center_slices)
        if not all(videos_exist):
            print(list(zip(external_videos, videos_exist)))
            raise FileExistsError("All avi files must exist in the main project; see 3a-alternate-only_make_videos.py")
    for ext_video, dlc_config in zip(external_videos, all_dlc_configs):
        i_neuron = _analyze_video_and_save_tracks(DEBUG, all_dfs, dlc_config, i_neuron, neuron2z_dict, [ext_video])
    final_df = _process_duplicates_to_final_df(all_dfs)

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
    df_fname = os.path.join(dest_folder, 'full_3d_tracks.h5')
    final_df.to_hdf(df_fname, "df_with_missing")

    fname = os.path.join(dest_folder, 'full_3d_tracks.csv')
    final_df.to_csv(fname)

    # Save in yaml
    udpates = {'final_3d_tracks': {'df_fname': df_fname}}
    edit_config(track_cfg['self_path'], udpates)

    return final_df


def _process_duplicates_to_final_df(all_dfs):
    # TODO: process repeats to create a final position
    final_df = pd.concat(all_dfs, axis=1)
    return final_df


def _analyze_video_and_save_tracks(DEBUG, all_dfs, dlc_config, i_neuron, neuron2z_dict, external_video_list=None):
    dlc_cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_config)
    if external_video_list[0] is None:
        video_list = list(dlc_cfg['video_sets'].keys())
        destfolder = None  # Save with videos
    else:
        video_list = [str(Path(vid).resolve()) for vid in external_video_list]
        destfolder = str(Path("3-tracking").resolve())  # Force a local save
    # Works even if already analyzed; skips if empty
    try:
        deeplabcut.analyze_videos(dlc_config, video_list, destfolder=destfolder)
    except IndexError:
        # Doesn't append anything to all_dfs
        print(f"No neurons found; skipping project {dlc_config}")
        return i_neuron
    # Get data for later use
    if destfolder is None:
        # i.e. it is saved where DLC expects it
        df_fname = get_annotations_from_dlc_config(dlc_config)
    else:
        df_fname = get_annotations_matching_video_in_folder(destfolder, video_list[0])
    if DEBUG:
        print(f"Using 2d annotations: {df_fname}")
    df = pd.read_hdf(df_fname)
    df_scorer = df.columns.values[0][0]
    df = df[df_scorer]
    # TODO: combine neurons that are the same
    new_names = df.columns.levels[0]  # Do NOT rename neurons
    # i_neuron_new = i_neuron + len(df.columns.levels[0])
    # neuron_range = range(i_neuron, i_neuron_new)
    # i_neuron = i_neuron_new
    # new_names = [f'neuron{i}' for i in neuron_range]
    # df.columns.set_levels(new_names, level=0, inplace=True)
    # Output (modified inplace)
    z = get_z_from_dlc_name(dlc_config)
    neuron2z_dict.update({n: z for n in new_names})
    all_dfs.append(df)
    return i_neuron


def make_all_dlc_labeled_videos(track_cfg, use_dlc_project_videos=True, DEBUG=False):
    """
    Applies DLC trained networks to the avi videos

    For visualization only
    """
    all_dlc_configs = track_cfg['dlc_projects']['all_configs']

    if use_dlc_project_videos:
        external_videos = [None for _ in all_dlc_configs]
    else:
        all_center_slices = track_cfg['training_data_2d']['all_center_slices']
        external_videos, videos_exist = _get_and_check_avi_filename(all_center_slices)
        if not all(videos_exist):
            print(list(zip(external_videos, videos_exist)))
            raise FileExistsError("All avi files must exist in the main project; see 3a-alternate-only_make_videos.py")

    for ext_video, dlc_config in zip(external_videos, all_dlc_configs):
        dlc_cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_config)
        if ext_video is None:
            video_list = list(dlc_cfg['video_sets'].keys())
            destfolder = None  # Save with videos
        else:
            video_list = [str(Path(ext_video).resolve())]
            destfolder = str(Path("3-tracking").resolve())  # Force a local save
            print(f"Checking for videos in {destfolder}")

        if not DEBUG:
            deeplabcut.create_labeled_video(dlc_config, video_list, destfolder=destfolder)
