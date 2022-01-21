import os
import re
from pathlib import Path
import deeplabcut
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions
from tqdm.auto import tqdm
from skimage import io
from skimage.util import img_as_ubyte
from DLC_for_WBFM.utils.feature_detection.custom_errors import AnalysisOutOfOrderError
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import build_subset_df_from_tracklets, build_subset_df_from_3dDLC, \
    build_dlc_annotation_from_tracklets, build_dlc_annotation_from_3dDLC
from DLC_for_WBFM.utils.visualization.visualize_using_dlc import save_dlc_annotations
from DLC_for_WBFM.utils.projects.project_config_classes import SubfolderConfigFile
from DLC_for_WBFM.utils.pipeline.paths_to_external_resources import get_pretrained_network_path

##
## Functions for building DLC projects using config class
##
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

    # Force shorter name
    dlc_opt = {'project': task_name[0] + label,
               'experimenter': experimenter[0],
               'videos': [video_path],
               'copy_videos': copy_videos,
               'working_directory': working_directory}

    dlc_config_fname = deeplabcut.create_new_project(**dlc_opt)

    if dlc_config_fname is None:
        print("Did not create dlc project... maybe it already exists?")
        print("If so, try deleting the project")
        raise FileExistsError

    return dlc_config_fname


def build_png_training_data(dlc_config, which_frames, verbose=0):
    """
    build_png_training_data_custom, but similar to the deeplabcut function

    see extract_frames() in deeplabcut
    """

    # Open video
    cfg = auxiliaryfunctions.read_config(dlc_config)
    videos = cfg.get("video_sets_original") or cfg["video_sets"]
    assert len(videos) == 1, "Only supports a single video"
    video = list(videos.keys())[0]

    # For some reason this import fails at the top of the script...
    from deeplabcut.utils.auxfun_videos import VideoReader
    cap = VideoReader(video)
    fname = Path(video)

    if not len(cap):
        print("Video could not be opened. Aborting...")
        raise FileNotFoundError

    # Now, extracting images
    output_path = Path(dlc_config).parents[0] / "labeled-data" / fname.stem
    full_imagenames = imsave_all_frames(cap, output_path, which_frames)
    dlc_dir = Path(dlc_config).parent
    relative_imagenames = [str(im.relative_to(dlc_dir)) for im in full_imagenames]
    if verbose >= 1:
        print(f"{relative_imagenames}")

    return relative_imagenames, output_path


def imsave_all_frames(cap, output_path, which_frames):
    all_img_names = []
    for i, index in enumerate(which_frames):
        cap.set_to_frame(index)  # extract a particular frame
        frame = cap.read_frame()
        if frame is not None:
            image = img_as_ubyte(frame)
            img_name = output_path.joinpath(f"img{i}.png")
            all_img_names.append(img_name)
            io.imsave(img_name, image)
    return all_img_names


# def OLD_training_data_from_tracklet_annotations(video_fname,
#                                             df_fname,
#                                             which_frames,
#                                             which_z,
#                                             dlc_config_fname,
#                                             max_z_dist_for_traces=2,
#                                             total_num_frames=500,
#                                             coord_names=None,
#                                             preprocessing_settings=None,
#                                             verbose=0):
#     """
#     Creates a set of training frames or volumes starting from a saved dataframe of tracklets
#         i.e. my custom dataframe format
#     Takes frames in the list which_frames, taking neurons that are present in each
#
#     Parameters
#     =================
#     video_fname: str
#         Path of the 2d avi video corresponding, from which frames will be taken
#     df_fname: str or pd.DataFrame
#         Path or object (DataFrame) with 3d annotations
#         Frame indices correspond to original btf
#     which_frames: list
#         List of indices that will be take from video_fname as training frames
#     which_z: int
#         Which z-slice corresponding to this video
#     dlc_config_fname: str
#         The location of the dlc config.yaml file
#         The pngs will be written in a subfolder of this parent
#     max_z_dist_for_traces: int
#         How many slices away a neuron centroid can be from which_z
#         i.e. dist <= which_z +- max_z_dist_for_traces
#     scorer: str
#         Name that will be written in the DLC dataframes
#     total_num_frames: int
#         Total number of frames in the video
#     coord_names: list of str
#         Which coordinates to save, in order. Options are:
#         ['x', 'y', 'z', 'likelihood']
#     preprocessing_settings: None or PreprocessingSettings object
#         If the training frames need to be preprocessed after being taken from the .avi video
#         Default is None
#     verbose: int
#         Amount to print
#
#     See also: training_data_3d_from_annotations
#     """
#
#     if coord_names is None:
#         coord_names = ['x', 'y', 'likelihood']
#
#     # Load the dataframe name, and produce DLC-style annotations
#     if type(df_fname) == str:
#         clust_df = pd.read_pickle(df_fname)
#     else:
#         assert type(df_fname) == pd.DataFrame, "Must pass dataframe or filename of dataframe"
#         clust_df = df_fname
#
#     # Build a sub-df with only the relevant neurons and slices
#     subset_opt = {'which_z': which_z,
#                   'max_z_dist': max_z_dist_for_traces,
#                   'verbose': 1}
#     subset_df = build_subset_df_from_tracklets(clust_df, which_frames, **subset_opt)
#     if len(subset_df) == 0:
#         print(f"Found no tracks long enough; aborting project: {dlc_config_fname}")
#         return None, None
#
#     # Save the individual png files
#     png_opt = {'dlc_config': dlc_config_fname,
#                'which_frames': which_frames}
#     out = build_png_training_data(**png_opt)
#     relative_imagenames, full_subfolder_name = out
#
#     # Cast the dataframe in DLC format
#     cfg = auxiliaryfunctions.read_config(dlc_config_fname)
#     options = {'min_length': 0,
#                'num_frames': total_num_frames,
#                'coord_names': coord_names,
#                'verbose': verbose,
#                'relative_imagenames': relative_imagenames,
#                'which_frame_subset': which_frames,
#                'scorer': cfg['scorer']}
#     new_dlc_df = build_dlc_annotation_from_tracklets(subset_df, **options)
#     if new_dlc_df is None:
#         print(f"Found no tracks long enough; aborting project: {dlc_config_fname}")
#         return None, None
#
#     # Save annotations using DLC-style names
#     data_dir = Path(dlc_config_fname).parent.joinpath('labeled-data')
#     img_subfolder_name = data_dir.joinpath(Path(video_fname).stem)
#     options = {'project_folder': str(img_subfolder_name)}
#     out = save_dlc_annotations(cfg['scorer'], df_fname, new_dlc_df, **options)
#     annotation_fname = out[1][1]
#
#     # Optional: plot the annotations on top of the frames
#     deeplabcut.check_labels(dlc_config_fname)
#
#     return new_dlc_df, annotation_fname


def training_data_from_3dDLC_annotations(video_fname,
                                         df_fname,
                                         which_frames,
                                         which_z,
                                         dlc_config_fname,
                                         max_z_dist_for_traces=2,
                                         total_num_frames=500,
                                         coord_names=None,
                                         preprocessing_settings=None,
                                         verbose=0):
    """
    Creates a set of training frames or volumes starting from a dataframe in 3d DLC format
    Takes all frames that are present in the dataframe

    Parameters
    =================
    video_fname: str
        Path of the 2d avi video corresponding, from which frames will be taken
    df_fname: str or pd.DataFrame
        Path or object (DataFrame) with 3d annotations
        Frame indices correspond to original btf
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
    if type(df_fname) == str:
        dlc3d_df: pd.DataFrame = pd.read_hdf(df_fname)
    else:
        assert type(df_fname) == pd.DataFrame, "Must pass dataframe or filename of dataframe"
        dlc3d_df: pd.DataFrame = df_fname

    # Build a sub-df only including neurons close in z
    subset_opt = {'which_z': which_z,
                  'max_z_dist': max_z_dist_for_traces,
                  'verbose': 1}
    subset_df = build_subset_df_from_3dDLC(dlc3d_df, **subset_opt)
    if len(subset_df) == 0:
        print(f"Found no tracks long enough; aborting project: {dlc_config_fname}")
        return None, None

    # TODO: Get which frames directly from dataframe?
    if len(which_frames) != len(list(dlc3d_df.index)):
        raise AnalysisOutOfOrderError(f"DataFrame subset in time")
    which_frames = list(dlc3d_df.index)
    # which_frames = list(subset_df.index)

    # Save the individual png files
    png_opt = {'dlc_config': dlc_config_fname,
               'which_frames': which_frames}
    out = build_png_training_data(**png_opt)
    relative_imagenames, full_subfolder_name = out

    # Cast the dataframe in DLC format
    cfg = auxiliaryfunctions.read_config(dlc_config_fname)
    options = {'min_length': 0,
               'num_frames': total_num_frames,
               'coord_names': coord_names,
               'verbose': verbose,
               'relative_imagenames': relative_imagenames,
               'which_frame_subset': which_frames,
               'scorer': cfg['scorer']}
    new_dlc_df = build_dlc_annotation_from_3dDLC(subset_df, **options)
    if new_dlc_df is None:
        print(f"Found no tracks long enough; aborting project: {dlc_config_fname}")
        return None, None

    # Save annotations using DLC-style names
    data_dir = Path(dlc_config_fname).parent.joinpath('labeled-data')
    img_subfolder_name = data_dir.joinpath(Path(video_fname).stem)
    options = {'project_folder': str(img_subfolder_name)}
    out = save_dlc_annotations(cfg['scorer'], df_fname, new_dlc_df, **options)
    annotation_fname = out[1][1]

    # Optional: plot the annotations on top of the frames
    deeplabcut.check_labels(dlc_config_fname)

    return new_dlc_df, annotation_fname


#
def update_pose_config(dlc_config_fname,
                       tracking_config: SubfolderConfigFile,
                       DEBUG=False):
    # Copied from: https://github.com/DeepLabCut/DeepLabCut/blob/master/examples/testscript.py
    cfg = auxiliaryfunctions.read_config(dlc_config_fname)

    posefile = posefile_from_dlc_cfg(cfg)

    # Copy settings from my global config file
    updates_from_project = tracking_config.config['pose_config_updates']

    pose_config = auxiliaryfunctions.read_plainconfig(posefile)
    # These are mostly from the official recommendations:
    # https://forum.image.sc/t/recommended-settings-for-tracking-fine-parts/36184/7
    pose_config['scale_jitter_lo'] = 0.75
    pose_config['scale_jitter_up'] = 1.25
    pose_config['augmentationprobability'] = 0.5
    # pose_config['batch_size']=8 #pick that as large as your GPU can handle it
    pose_config['elastic_transform'] = True
    pose_config['rotation'] = updates_from_project.get('rotation', 180)
    pose_config['covering'] = True
    pose_config['motion_blur'] = True
    pose_config['optimizer'] = "adam"
    pose_config['dataset_type'] = 'imgaug'
    # My changes and additions
    pose_config['multi_step'] = updates_from_project.get('multi_step', None)
    pose_config['save_iters'] = 10000
    if DEBUG:
        pose_config['multi_step'] = [[0.005, 1000]]
        pose_config['save_iters'] = 900
    elif pose_config['multi_step'] is None:
        pose_config['multi_step'] = [[5e-4, 5e3],
                                     [1e-4, 1e4],
                                     [5e-5, 3e4],
                                     [1e-5, 5e4]]
    pose_config['pos_dist_thresh'] = updates_from_project.get('pos_dist_thresh', 9)  # We have very small objects
    # pose_config['pairwise_predict'] = False  # Broken?

    # Reuse initial weights, and decrease the training time
    if tracking_config.config.get('use_pretrained_dlc', False):
        num_neurons = len(cfg['bodyparts'])
        network_path = get_pretrained_network_path(num_neurons)

        if Path(network_path + ".meta").exists():
            print(f"Using pretrained network at {network_path}")
            pose_config['init_weights'] = network_path

            # steps = pose_config['multi_step']
            # steps = steps[-1]
            # steps[1] = 5e4  # Shorten
            steps = [[0.00005, 10000],
                     [0.00001, 30000]]
            print(f"Shortening training to: {steps}")
            pose_config['multi_step'] = steps

        else:
            print(f"No pretrained network exists for {num_neurons} neurons ({network_path})")
    else:
        print(f"Training project {dlc_config_fname} from scratch")

    print(f"Updates: {pose_config}")
    auxiliaryfunctions.write_plainconfig(posefile, pose_config)


def update_all_pose_configs(tracking_config: SubfolderConfigFile,
                            updates: dict =None):
    """

    Updates all DLC pose config files in a project

    Parameters
    ----------
    tracking_config
    updates - custom update dictionary; if none, then use the default update from tracking_config.yaml

    Returns
    -------
    None

    See also: update_pose_config
    """

    all_dlc_configs = tracking_config.config['dlc_projects']['all_configs']

    for dlc_config_fname in tqdm(all_dlc_configs):

        if updates is None:
            # Use the defaults
            update_pose_config(dlc_config_fname, tracking_config)
        else:
            cfg = auxiliaryfunctions.read_config(dlc_config_fname)
            posefile = posefile_from_dlc_cfg(cfg)
            pose_config = auxiliaryfunctions.read_plainconfig(posefile)
            pose_config.update(updates)

            auxiliaryfunctions.write_plainconfig(posefile, pose_config)

    return None


def posefile_from_dlc_cfg(cfg):
    posefile = os.path.join(
        cfg["project_path"],
        "dlc-models/iteration-"
        + str(cfg["iteration"])
        + "/"
        + cfg["Task"]
        + cfg["date"]
        + "-trainset"
        + str(int(cfg["TrainingFraction"][0] * 100))
        + "shuffle"
        + str(1),
        "train/pose_cfg.yaml",
    )

    return posefile


##
## Helper functions
##


def get_z_from_dlc_name(name):
    regex = r"c[\d]+"
    results = re.findall(regex, name, re.MULTILINE)
    return [int(s[1:]) for s in results][0]


def get_annotations_from_dlc_config(dlc_config, use_filtered=True):
    """
    Get the .h5 file corresponding to the DLC tracks

    Assumes it is the same folder as the video itself, i.e. the videos/ folder in the DLC project

    If there is more than one .h5 file, uses use_filtered to choose which to use
        (assuming the additional files come from built-in DLC filtering functions)
    """
    video_dir = Path(dlc_config).with_name('videos')
    fnames = os.listdir(video_dir)
    annotation_names = [f for f in fnames if f.endswith('.h5')]
    if len(annotation_names) > 1:
        print("Using filtered annotations")
        if use_filtered:
            annotation_is_filtered = [('filtered' in f) for f in annotation_names]
            which_annotation_inds = np.where(annotation_is_filtered)[0]
            annotation_names = [annotation_names[i] for i in which_annotation_inds]
            if len(which_annotation_inds) > 0:
                print(f"Found more than one filtered annotation for {dlc_config}; taking first one")
        else:
            annotation_is_not_filtered = [('filtered' not in f) for f in annotation_names]
            which_annotation_inds = np.where(annotation_is_not_filtered)[0]
            annotation_names = [annotation_names[i] for i in which_annotation_inds]
            if len(which_annotation_inds) > 0:
                print(f"Found more than one non-filtered annotation for {dlc_config}; taking first one")
    # No matter what, we take the first one
    final_name = annotation_names[0]
    print(f"Using found annotations: {final_name}")
    return Path(video_dir).joinpath(final_name)


def get_annotations_matching_video_in_folder(annotation_dir, video_fname):
    fnames = os.listdir(annotation_dir)
    video_stem = Path(video_fname).stem  # TODO: can cause errors if center1 and center10 both exist
    annotation_names = [f for f in fnames if (f.startswith(video_stem) and f.endswith('.h5'))]
    if len(annotation_names) > 1:
        print(f"Found more than one annotation for {video_fname}; taking first one")
    elif len(annotation_names) == 0:
        raise FileNotFoundError(f"Found no annotations for video {video_fname}")
    annotation_names = annotation_names[0]
    print(f"Using found annotations: {annotation_names}")
    return Path(annotation_dir).joinpath(annotation_names)
