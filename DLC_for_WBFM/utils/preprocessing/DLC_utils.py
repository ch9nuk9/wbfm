import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
# from deeplabcut.generate_training_dataset.frame_extraction import extract_frames

from DLC_for_WBFM.utils.preprocessing.convert_matlab_annotations_to_DLC import csv_annotations2config_names
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings, perform_preprocessing
from DLC_for_WBFM.bin.configuration_definition import load_config, DLCForWBFMTracking, save_config
from DLC_for_WBFM.utils.video_and_data_conversion.video_conversion_utils import write_numpy_as_avi
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_coords3d
from DLC_for_WBFM.utils.feature_detection.visualize_using_dlc import build_subset_df, build_dlc_annotation_all, build_relative_imagenames, save_dlc_annotations
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import best_tracklet_covering
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
import pandas as pd
import numpy as np
import tifffile
from pathlib import Path
# import pims
# import PIL
from skimage import io
from skimage.util import img_as_ubyte
import os
from tqdm import tqdm


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
               'experimenter':c.experimenter[:4],
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

    # Force shorter name
    dlc_opt = {'project':task_name[0] + label,
               'experimenter':experimenter[0],
               'videos':[video_path],
               'copy_videos':copy_videos,
               'working_directory':working_directory}

    dlc_config_fname = deeplabcut.create_new_project(**dlc_opt)

    if dlc_config_fname is None:
        print("Did not create dlc project... maybe it already exists?")
        print("If so, try deleting the project")
        raise ValueError

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

    # Get the filenames to match with old api
    # dlc_folder = Path(dlc_config).parent
    # full_subfolder_name = list(dlc_folder.iterdir())
    # assert len(full_subfolder_name)==1, "Found more than one subfolder..."
    # full_subfolder_name = full_subfolder_name[0]
    # relative_imagenames = list(relative_imagenames.iterdir())
    # assert len(relative_imagenames)==len(which_frames)
    # full_subfolder_name = str(full_subfolder_name)
    # relative_imagenames = [str(im) for im in relative_imagenames]
    if verbose >= 1:
        # print(f"Extracted images in subfolder {full_subfolder_name}:")
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
            # img_name = (
            #     str(output_path)
            #     + "/img"
            #     + str(index).zfill(indexlength)
            #     + ".png"
            # )
            io.imsave(img_name, image)
    return all_img_names

# def build_png_training_data_custom(dlc_config,
#                             video_fname,
#                             which_frames,
#                             verbose=0):
#     """
#     Extracts a series of pngs from a full video (.avi)
#
#
#     See also: build_tif_training_data
#     """
#
#     # Get the file names
#     name_opt = {'file_ext': 'avi', 'num_frames': len(which_frames)}
#     out = build_relative_imagenames(video_fname, **name_opt)
#     relative_imagenames, subfolder_name = out
#
#     # Initilize the training data subfolder
#     dlc_config = auxiliaryfunctions.read_config(dlc_config)
#     project_folder = dlc_config['project_path']
#     full_subfolder_name = os.path.join(project_folder, subfolder_name)
#     if not os.path.isdir(full_subfolder_name):
#         os.mkdir(full_subfolder_name)
#
#     # Write the png files
#     with pims.PyAVVideoReader(video_fname) as video_reader:
#         if verbose >= 1:
#             print('Writing png files...')
#         for i, rel_fname in tqdm(zip(which_frames, relative_imagenames), total=len(which_frames)):
#             dat = video_reader[i]
#             fname = os.path.join(project_folder, rel_fname)
#             skio.imsave(fname, dat)
#
#     if verbose >= 1:
#         print(f"{len(which_frames)} png files written in project {full_subfolder_name}")
#
#     return relative_imagenames, full_subfolder_name


def training_data_from_annotations(vid_fname,
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
    if type(df_fname) == str:
        clust_df = pd.read_pickle(df_fname)
    else:
        assert type(df_fname) == pd.DataFrame, "Must pass dataframe or filename of dataframe"
        clust_df = df_fname

    # Build a sub-df with only the relevant neurons and slices
    subset_opt = {'which_z': which_z,
                  'max_z_dist': max_z_dist_for_traces,
                  'verbose': 1}
    subset_df = build_subset_df(clust_df, which_frames, **subset_opt)

    # TODO: Save the individual png files
    png_opt = {'dlc_config': dlc_config_fname,
               'which_frames': which_frames}
    out = build_png_training_data(**png_opt)
    relative_imagenames, full_subfolder_name = out

    # Cast the dataframe in DLC format
    cfg = auxiliaryfunctions.read_config(dlc_config_fname)
    opt = {'min_length': 0,
           'num_frames': total_num_frames,
           'coord_names': coord_names,
           'verbose': verbose,
           'relative_imagenames': relative_imagenames,
           'which_frame_subset': which_frames,
           'scorer': cfg['scorer']}
    new_dlc_df = build_dlc_annotation_all(subset_df, **opt)
    if new_dlc_df is None:
        print("Found no tracks long enough; aborting")
        return None

    # Save annotations using DLC-style names
    data_dir = Path(dlc_config_fname).parent.joinpath('labeled-data')
    img_subfolder_name = data_dir.joinpath(Path(vid_fname).stem)
    opt = {'project_folder': str(img_subfolder_name)}
    out = save_dlc_annotations(cfg['scorer'], df_fname, new_dlc_df, **opt)
    annotation_fname = out[1][1]

    # Optional: plot the annotations on top of the frames
    deeplabcut.check_labels(dlc_config_fname)

    return new_dlc_df, annotation_fname


#
def update_pose_config(dlc_config_fname, DEBUG=False):
    # Copied from: https://github.com/DeepLabCut/DeepLabCut/blob/master/examples/testscript.py
    cfg = auxiliaryfunctions.read_config(dlc_config_fname)

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

    pose_config = auxiliaryfunctions.read_plainconfig(posefile)
    # These are mostly from the official recommendations:
    # https://forum.image.sc/t/recommended-settings-for-tracking-fine-parts/36184/7
    pose_config['scale_jitter_lo'] = 0.75
    pose_config['scale_jitter_up'] = 1.25
    pose_config['augmentationprobability'] = 0.5
    # pose_config['batch_size']=8 #pick that as large as your GPU can handle it
    pose_config['elastic_transform'] = True
    pose_config['rotation'] = 180
    pose_config['covering'] = True
    pose_config['motion_blur'] = True
    pose_config['optimizer'] = "adam"
    pose_config['dataset_type'] = 'imgaug'
    # My changes and additions
    if DEBUG:
        pose_config['multi_step'] = [[0.005, 1000]]
        pose_config['save_iters'] = 900
    else:
        pose_config['multi_step'] = [[5e-4, 5e3],
                                     [1e-4, 1e4],
                                     [5e-5, 3e4],
                                     [1e-5, 5e4]]
        # pose_config['multi_step'] = [[0.005, 7500], [5e-4, 2e4], [1e-4, 5e4]]
        pose_config['save_iters'] = 10000
    pose_config['pos_dist_thresh'] = 13  # 15  # We have very small objects
    pose_config['pairwise_predict'] = False  # Broken?

    auxiliaryfunctions.write_plainconfig(posefile, pose_config)


###
### For use with training a stack of DLC (step 3 of pipeline)
###

def create_dlc_training_from_tracklets(vid_fname,
                                       config,
                                       scorer=None,
                                       task_name=None,
                                       verbose=0,
                                       DEBUG=False):

    ########################
    # Load annotations
    ########################
    df_fname = config['training_data_3d']['annotation_fname']
    df = pd.read_pickle(df_fname)

    ########################
    # Prepare for dlc-style training data
    ########################

    # Choose a subset of frames with enough tracklets
    num_frames_needed = config['training_data_3d']['num_training_frames']
    tracklet_opt = {'num_frames_needed': num_frames_needed,
                    'num_frames': config['dataset_params']['num_frames'],
                    'verbose': 1}
    if DEBUG:
        tracklet_opt['num_frames_needed'] = 2
    which_frames = best_tracklet_covering(df, **tracklet_opt)
    # Also save these chosen frames
    updates = {'which_frames': which_frames}
    config['training_data_3d'].update(updates)
    edit_config(config['self_path'], config)

    all_center_slices = config['training_data_2d']['all_center_slices']
    if DEBUG:
        all_center_slices = [all_center_slices[0]]

    ########################
    # Get or make the video
    ########################
    # TODO: for now, requires re-preprocessing

    def make_avi_name(center):
        fname = f"center{center}.avi"  # NOT >8 CHAR (without .avi)
        if len(fname) > 12:
            # TODO: fix this bug
            # Another function clips labeled-data/folder-name at 8 chars
            # But, that name must be the same as the video
            raise ValueError(f"Bug if this is too long {fname}")
        return fname
    # FIRST: check if we actually need to rewrite the videos
    video_exists = []
    all_avi_fnames = []
    for center in all_center_slices:
        # Make minimax video from btf
        this_avi_fname = make_avi_name(center)
        all_avi_fnames.append(this_avi_fname)
        if os.path.exists(this_avi_fname):
            print(f"Using video at: {this_avi_fname}")
            video_exists.append(True)
        else:
            video_exists.append(False)
            # write_numpy_as_avi(preprocessed_dat[:, center, ...], **vid_opt)
            # write_video_projection_from_ome_file_subset(**vid_opt)

    # IF videos are required, then prep the data
    if all(video_exists):
        print("All required videos exist; no preprocessing necessary")
        preprocessed_dat = []
    else:
        with tifffile.TiffFile(vid_fname) as tif:
            sz = tif.pages[0].shape
        vid_opt = {'fps': config['dataset_params']['fps'],
                   'frame_height': sz[0],
                   'frame_width': sz[1]}
        if verbose >= 1:
            print("Preprocessing data, this could take a while...")
        p = PreprocessingSettings.load_from_yaml(config['preprocessing_config'])
        start_volume = config['dataset_params']['start_volume']
        num_total_frames = start_volume + config['dataset_params']['num_frames']
        num_slices = config['dataset_params']['num_slices']
        if DEBUG:
            # Make a much shorter video
            num_total_frames = which_frames[-1] + 1
        preprocessed_dat = np.zeros((num_total_frames, num_slices) + sz)

        # Load data and preprocess
        for i in tqdm(list(range(num_total_frames))):
            dat_raw = get_single_volume(vid_fname, i, num_slices, dtype='uint16')
            # Don't preprocess data that we didn't even segment!
            if i >= start_volume:
                preprocessed_dat[i, ...] = perform_preprocessing(dat_raw, p)
            else:
                preprocessed_dat[i, ...] = dat_raw

    ########################
    # Initialize the DLC projects
    ########################
    # Get dlc project and naming options
    dlc_opt = {'task_name': task_name,
               'experimenter': scorer,
               'working_directory': '3-tracking',
               'copy_videos': True}
    # Get a few frames as training data
    png_opt = {}
    png_opt['df_fname'] = df
    # png_opt['scorer'] = scorer
    png_opt['total_num_frames'] = config['dataset_params']['num_frames']
    png_opt['coord_names'] = ['x', 'y']
    png_opt['which_frames'] = config['training_data_3d']['which_frames']
    png_opt['max_z_dist_for_traces'] = config['training_data_2d']['max_z_dist_for_traces']
    # Connecting these frames to a network architecture
    net_opt = {'net_type': "resnet_50",  # 'mobilenet_v2_0.35' #'resnet_50
               'augmenter_type': "imgaug"}
    # pose_opt = {}  # TODO
    # Actually make projects
    all_dlc_configs = []
    for i, center in enumerate(all_center_slices):
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

    # Then delete the created avis because they are copied into the DLC folder
    # [os.remove(f) for f in all_avi_fnames]

    # Save list of dlc config names
    # Make names relative to project folder
    # prj = Path(config['project_path']).parent
    # all_dlc_configs = [str(Path(cfg).relative_to(prj)) for cfg in all_dlc_configs]
    config['dlc_projects']['all_configs'] = all_dlc_configs
    edit_config(config['self_path'], config)


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
        # deeplabcut.train_network(dlc_config)
        try:
            deeplabcut.train_network(dlc_config)
        except CancelledError:
            # This means it finished the planned number of steps
            pass
