import os
from concurrent import futures
from pathlib import Path
from typing import Tuple, List

import deeplabcut
import numpy as np
import pandas as pd
import zarr
from scipy.spatial.distance import pdist
from tqdm import tqdm

from DLC_for_WBFM.utils.preprocessing.DLC_utils import get_annotations_from_dlc_config, get_z_from_dlc_name, \
    update_pose_config, training_data_from_tracklet_annotations, \
    create_dlc_project, get_annotations_matching_video_in_folder, training_data_from_3dDLC_annotations
from DLC_for_WBFM.utils.preprocessing.convert_matlab_annotations_to_DLC import csv_annotations2config_names
from DLC_for_WBFM.utils.preprocessing.utils_tif import _get_video_options
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config, config_file_with_project_context
from DLC_for_WBFM.utils.projects.utils_project import edit_config, safe_cd
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import best_tracklet_covering_from_my_matches, \
    get_or_recalculate_which_frames, calculate_best_covering_from_tracklets
from DLC_for_WBFM.utils.video_and_data_conversion.video_conversion_utils import write_numpy_as_avi


###
### For use with training a stack of DLC (step 3 of pipeline)
###

def create_only_videos(vid_fname, config, verbose=1, DEBUG=False):
    """
    Shortened version of create_dlc_training_from_tracklets() that only creates the videos

    Does not require that training data is present; intended to be used when reusing other networks
    """

    all_center_slices = config['training_data_2d']['all_center_slices']
    all_avi_fnames, preprocessed_dat, vid_opt, video_exists = _prep_videos_for_dlc(all_center_slices, config, vid_fname)

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


def create_dlc_training_from_tracklets(project_config: modular_project_config,
                                       training_config: config_file_with_project_context,
                                       tracking_config: config_file_with_project_context,
                                       scorer: str = None,
                                       task_name: str = None,
                                       DEBUG: bool = False) -> None:


    df_fname = training_config.resolve_relative_path('df_raw_3d_tracks')
    if df_fname.endswith(".pickle"):
        raise DeprecationWarning("Creating training data from raw pickle not supported; convert to 3d DLC dataframe")
        # df = pd.read_pickle(df_fname)
    else:
        assert df_fname.endswith(".h5")
        df = pd.read_hdf(df_fname)

    all_center_slices, which_frames = _get_frames_for_dlc_training(DEBUG, df, tracking_config)
    # edit_config(config['self_path'], config)
    tracking_config.update_on_disk()

    vid_cfg = tracking_config.config
    vid_cfg['dataset_params'] = project_config.config['dataset_params']
    vid_fname = project_config.config['preprocessed_red']
    all_avi_fnames, preprocessed_dat, vid_opt, video_exists = _prep_videos_for_dlc(all_center_slices, vid_cfg, vid_fname)

    dlc_opt, net_opt, png_opt = _define_project_options(vid_cfg, df, scorer, task_name)
    # Actually make projects
    # all_dlc_configs = []
    with safe_cd(project_config.project_dir):
        for i, center in enumerate(all_center_slices):
            try:
                this_dlc_config = _initialize_project_from_btf(all_avi_fnames, center, dlc_opt, i, net_opt, png_opt,
                                             preprocessed_dat, vid_opt, video_exists, tracking_config.config)
                # all_dlc_configs.append(this_dlc_config)
            except FileExistsError:
                print("Found existing folder, skipping")

    # def parallel_func(i_center):
    #     i, center = i_center
    #     _initialize_project_from_btf(all_avi_fnames, center, dlc_opt, i, net_opt, png_opt,
    #                                  preprocessed_dat, vid_opt, video_exists, config)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_center_slices)) as executor:
    #     futures = {executor.submit(parallel_func, i): i for i in enumerate(all_center_slices)}
    #     all_dlc_configs = [f.result() for f in concurrent.futures.as_completed(futures)]
        # futures = executor.map(parallel_func, enumerate(all_center_slices))
        # all_dlc_configs = [f.result() for f in futures]

    # Then delete the created avis because they are copied into the DLC folder
    # [os.remove(f) for f in all_avi_fnames]

    # Save list of dlc config names
    all_dlc_configs = []
    base_dir = Path(os.path.join(project_config.project_dir, '3-tracking'))
    for fname in tqdm(base_dir.iterdir()):
        if fname.is_dir():
            # Check for DLC project
            dlc_name = fname.joinpath('config.yaml')
            if dlc_name.exists():
                all_dlc_configs.append(str(dlc_name))
    print(f"Found config files: {all_dlc_configs}")

    tracking_config.config['dlc_projects']['all_configs'] = all_dlc_configs
    tracking_config.update_on_disk()
    # edit_config(config['self_path'], config)


def _prep_videos_for_dlc(all_center_slices: List[int], config: dict,
                         vid_fname: str) -> Tuple[List[str], zarr.Array, dict, List[bool]]:
    all_avi_fnames, video_exists = _get_and_check_avi_filename(all_center_slices, subfolder="3-tracking")
    # IF videos are required, then prep the data
    _, vid_opt = _get_video_options(config, vid_fname)
    if all(video_exists):
        print("All required videos exist; no preprocessing necessary")
        preprocessed_dat = []
    else:
        # DEPRECATE PREPROCESSING
        # preprocessed_dat, vid_opt = preprocess_all_frames_using_config(DEBUG, config, verbose, vid_fname, which_frames)
        preprocessed_dat = zarr.open(vid_fname)
    return all_avi_fnames, preprocessed_dat, vid_opt, video_exists


def _get_and_check_avi_filename(all_center_slices: List[int],
                                subfolder: str = "3-tracking") -> Tuple[List[str], List[bool]]:
    """Returns relative path of avi file, not just name"""
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


def _define_project_options(config: dict, df: pd.DataFrame, scorer: str, task_name: str) -> Tuple[dict, dict, dict]:
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
                                 preprocessed_dat, vid_opt, video_exists, tracking_config):
    this_avi_fname = _get_or_make_avi(all_avi_fnames, center, i, preprocessed_dat, vid_opt, video_exists)
    # Make dlc project
    dlc_opt['label'] = f"-c{center}"
    dlc_opt['video_path'] = this_avi_fname
    this_dlc_config = create_dlc_project(**dlc_opt)
    # Training frame extraction
    png_opt['which_z'] = center
    png_opt['dlc_config_fname'] = this_dlc_config
    png_opt['vid_fname'] = this_avi_fname
    ann_fname = training_data_from_3dDLC_annotations(**png_opt)[1]
    if ann_fname is not None:
        # Synchronize the dlc_config with the annotations
        csv_annotations2config_names(this_dlc_config, ann_fname, num_dims=2, to_add_skeleton=True)
        # Format the training data
        deeplabcut.create_training_dataset(this_dlc_config, **net_opt)
        update_pose_config(this_dlc_config, tracking_config)
        # Save to list
        return this_dlc_config
    else:
        return None


def _get_or_make_avi(all_avi_fnames, center, i, preprocessed_dat, vid_opt, video_exists):
    """
    Note: all_avi_fnames should have the relative path, not just the name
    """
    # Make or get video
    this_avi_fname = all_avi_fnames[i]
    if not video_exists[i]:
        vid_opt['out_fname'] = this_avi_fname
        # If color, use adjacent planes as color information
        if not vid_opt.get('is_color', False):
            avi_data = preprocessed_dat[:, center, ...]
        else:
            z_range = np.clip([center-1, center, center+1], 0, preprocessed_dat.shape[1])
            avi_data = np.stack([preprocessed_dat[:, i, ...] for i in z_range], axis=-1)

        write_numpy_as_avi(avi_data, **vid_opt)
    return this_avi_fname


def _get_frames_for_dlc_training(DEBUG: bool, df: pd.DataFrame,
                                 tracking_config: config_file_with_project_context):
    # Choose a subset of frames with enough tracklets
    which_frames = tracking_config.config['training_data_3d'].get('which_frames', None)

    if which_frames is None:
        num_training_frames = tracking_config.config['training_data_3d']['num_training_frames']
        which_frames = calculate_best_covering_from_tracklets(df, num_training_frames)
        # which_frames = get_or_recalculate_which_frames(DEBUG, df, num_frames, tracking_config)
        # raise DeprecationWarning("Calculating which frames at this point is deprecated; calculate before calling this")
        # num_frames_needed = config['training_data_3d']['num_training_frames']
        # tracklet_opt = {'num_frames_needed': num_frames_needed,
        #                 'num_frames': config['dataset_params']['num_frames'],
        #                 'verbose': 1}
        # if DEBUG:
        #     tracklet_opt['num_frames_needed'] = 2
        # which_frames, _ = best_tracklet_covering(df, **tracklet_opt)
    # else:
    #     which_frames = config['training_data_3d']['which_frames']
    #     if which_frames is None:
    #         raise DeprecationWarning(
    #             "Calculating which frames at this point is deprecated; calculate before calling this")
    # Also save these chosen frames
    updates = {'which_frames': which_frames}
    tracking_config.config['training_data_3d'].update(updates)

    all_center_slices = tracking_config.config['training_data_2d']['all_center_slices']
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


def train_all_dlc_from_config(config: dict) -> None:
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


def make_3d_tracks_from_stack(track_cfg: dict, use_dlc_project_videos: bool = True,
                              DEBUG: bool = False) -> pd.DataFrame:
    """
    Applies trained DLC networks to full 2d videos and collects into 3d track

    Can be used with the videos from the DLC projects, or external ones
    """

    all_dlc_configs = track_cfg['dlc_projects']['all_configs']
    use_filtered = track_cfg['final_3d_tracks'].get('use_filtered', False)

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
        i_neuron = _analyze_video_and_save_tracks(DEBUG, all_dfs, dlc_config, i_neuron, neuron2z_dict,
                                                  use_filtered, [ext_video])
    final_df = _process_duplicates_to_final_df(all_dfs)

    # Collect 2d data
    # i.e. just add the z coordinate to it
    # For some reason, the concat after adding z was broken
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

    # Save only df_fname in yaml; don't overwrite other fields
    updates = track_cfg['final_3d_tracks']
    updates['df_fname'] = df_fname
    edit_config(track_cfg['self_path'], {'final_3d_tracks': updates})

    return final_df


def _process_duplicates_to_final_df(all_dfs: List[pd.DataFrame], verbose: bool = 0) -> pd.DataFrame:
    # Do a naive concatenation, and check for duplicates
    df_with_duplicates = pd.concat(all_dfs, axis=1)
    duplicate_ind = np.where(df_with_duplicates.columns.duplicated(keep=False))[0]
    duplicate_names = [df_with_duplicates.iloc[:, i].name[0] for i in duplicate_ind]
    duplicate_names = list(set(duplicate_names))

    # Use heuristics to combine multiple points of varying confidence levels
    all_dfs_to_concat = []
    for name in tqdm(duplicate_names):
        this_df = df_with_duplicates[name]
        if verbose >= 2:
            print(f"Found {len(this_df.columns) // 3} duplicates for neuron {name}")
        new_xy_conf = consolidate_duplicates(this_df, verbose=verbose-2)
        # Align with DLC formatting
        xy_conf_dict = {
            (name, 'x'): new_xy_conf[:, 0],
            (name, 'y'): new_xy_conf[:, 1],
            (name, 'likelihood'): new_xy_conf[:, 2],
        }
        columns = pd.MultiIndex.from_tuples(xy_conf_dict.keys(), names=["bodyparts", "coords"])
        tmp_df = pd.DataFrame(xy_conf_dict, columns=columns)
        all_dfs_to_concat.append(tmp_df)
    consolidated_df = pd.concat(all_dfs_to_concat, axis=1)

    # Combine with neurons that were unique
    unique_ind = np.where(~df_with_duplicates.columns.duplicated(keep=False))[0]
    df_without_duplicates = df_with_duplicates.iloc[:, unique_ind]

    final_df = pd.concat([consolidated_df, df_without_duplicates], axis=1)

    return final_df


def _analyze_video_and_save_tracks(DEBUG: bool, all_dfs: List[pd.DataFrame], dlc_config: dict, i_neuron: int,
                                   neuron2z_dict: dict,
                                   use_filtered: bool = False, external_video_list: list = None) -> int:
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
        df_fname = get_annotations_from_dlc_config(dlc_config, use_filtered=use_filtered)
    else:
        df_fname = get_annotations_matching_video_in_folder(destfolder, video_list[0])
    if DEBUG:
        print(f"Using 2d annotations: {df_fname}")
    df: pd.DataFrame = pd.read_hdf(df_fname)
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
        external_videos = _get_dlc_video_names_from_config(track_cfg)

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


def _get_dlc_video_names_from_config(track_cfg):
    all_center_slices = track_cfg['training_data_2d']['all_center_slices']
    external_videos, videos_exist = _get_and_check_avi_filename(all_center_slices)
    if not all(videos_exist):
        print(list(zip(external_videos, videos_exist)))
        raise FileExistsError("All avi files must exist in the main project; see 3a-alternate-only_make_videos.py")
    return external_videos


def filter_all_dlc_tracks(track_cfg, filter_mode='arima', use_dlc_project_videos=True, DEBUG=False):
    """
    Applies a simple median, arima, or spline filter to each DLC project

    See also: https://github.com/DeepLabCut/DeepLabCut/blob/6c3df66e901874371d2339bbccd968a07230fc77/deeplabcut/post_processing/filtering.py
    """
    all_dlc_configs = track_cfg['dlc_projects']['all_configs']

    if use_dlc_project_videos:
        external_videos = [None for _ in all_dlc_configs]
    else:
        external_videos = _get_dlc_video_names_from_config(track_cfg)

    def parallel_func(vid_and_cfg, DEBUG=DEBUG, filter_mode=filter_mode):
    # for ext_video, dlc_config in zip(external_videos, all_dlc_configs):
        ext_video, dlc_config = vid_and_cfg
        dlc_cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_config)
        if ext_video is None:
            video_list = list(dlc_cfg['video_sets'].keys())
        else:
            video_list = [str(Path(ext_video).resolve())]

        if not DEBUG:
            deeplabcut.filterpredictions(dlc_config, video_list, filtertype=filter_mode)

    with futures.ThreadPoolExecutor(max_workers=32) as executor:
        results_iter = executor.map(parallel_func, zip(external_videos, all_dlc_configs))
        results = [result for result in results_iter]



##
## Functions for consolidating multiple 2d tracks into one
##

# Loop through the duplicates and apply heuristics
def is_finalized(pts):
    return pts.shape[0] == 1 or pts.ndim == 1


def consolidate_row(this_row, verbose=0):
    """
    Heuristics for combining multiple points into one final position

    Algorithm:
    Given multiple tracked points, each with XY coordinates and likelihood:
    if there is only one confident point:
       keep that
    if the points are close and the confidences are > THRESH:
       average the points
    if the points are far and one is MUCH BETTER than the other:
       remove the low confidence point
    if no confidences are high:
       NaN the point (no need to do anything; low confidence will be removed later)
    if multiple different locations are high confidence:
       vote and average... if it is close, NaN the point
    """

    pts = np.reshape(np.array(this_row), (-1, 3))

    # Remove nan
    pts = pts[~np.isnan(pts[:, 2]), :]
    if is_finalized(pts):
        if verbose >= 1:
            print("Only one non-nan point")
        return pts

    # Don't even consider below this
    min_confidence_threshold = 0.4
    pts = pts[pts[:, 2] > min_confidence_threshold, :]
    if len(pts) == 0:
        if verbose >= 1:
            print("No good points; returning 0")
        return np.array([0.0, 0.0, 0.0])
    if is_finalized(pts):
        if verbose >= 1:
            print("Only one remaining good point")
        return pts

    # Average if the distances are close
    averaging_distance_threshold = 5.0

    pts_dists = pdist(pts[:, :2])
    if all(pts_dists < averaging_distance_threshold):
        pts = np.mean(pts, axis=0)
    if is_finalized(pts):
        if verbose >= 1:
            print("Averaged good, close points")
        return pts

    # If multiple high-confidence points, try to vote
    # Requires at least 3 points
    if pts.shape[0] < 3:
        if verbose >= 1:
            print("2 points but both have high confidence... returning zero confidence")
        return np.array([0.0, 0.0, 0.0])
    else:
        if verbose >= 1:
            print("Trying to cluster... TODO")
            print(pts.shape)
        return np.array([0.0, 0.0, 0.0])


def consolidate_duplicates(df, verbose=0):
    final_xyconf = []
    for _, row in df.iterrows():
        new_row = np.squeeze(consolidate_row(row, verbose=verbose - 2))
        if verbose >= 2:
            print(new_row)
        final_xyconf.append(new_row)
    return np.vstack(final_xyconf)
