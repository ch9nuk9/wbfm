import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
from DLC_for_WBFM.config.class_configuration import load_config, DLCForWBFMTracking, save_config
from DLC_for_WBFM.utils.feature_detection.visualize_using_dlc import build_subset_df, build_dlc_annotation_all, save_dlc_annotations
from DLC_for_WBFM.utils.projects.utils_project import load_config
import pandas as pd
from pathlib import Path
import re
from skimage import io
from skimage.util import img_as_ubyte
import os


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
    dlc_opt = {'project': task_name[0] + label,
               'experimenter': experimenter[0],
               'videos': [video_path],
               'copy_videos': copy_videos,
               'working_directory': working_directory}

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
    if len(subset_df) == 0:
        print(f"Found no tracks long enough; aborting project: {dlc_config_fname}")
        return None, None

    # Save the individual png files
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
        print(f"Found no tracks long enough; aborting project: {dlc_config_fname}")
        return None, None

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
def update_pose_config(dlc_config_fname, project_config, DEBUG=False):
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

    updates_from_project = project_config['tracking_config']['pose_config_updates']

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
        # pose_config['multi_step'] = [[0.005, 7500], [5e-4, 2e4], [1e-4, 5e4]]
    pose_config['pos_dist_thresh'] = updates_from_project.get('pos_dist_thresh', 13)  # We have very small objects
    # pose_config['pairwise_predict'] = False  # Broken?

    auxiliaryfunctions.write_plainconfig(posefile, pose_config)


##
## Helper functions
##


def get_z_from_dlc_name(name):
    regex = r"c[\d]+"
    results = re.findall(regex, name, re.MULTILINE)
    return [int(s[1:]) for s in results][0]


def get_annotations_from_dlc_config(dlc_config):
    video_dir = Path(dlc_config).with_name('videos')
    fnames = os.listdir(video_dir)
    annotation_names = [f for f in fnames if f.endswith('.h5')]
    if len(annotation_names) > 1:
        print(f"Found more than one annotation for {dlc_config}; taking first one")
    annotation_names = annotation_names[0]
    print(f"Using found annotations: {annotation_names}")
    return Path(video_dir).joinpath(annotation_names)


def get_annotations_matching_video_in_folder(annotation_dir, video_fname):
    fnames = os.listdir(annotation_dir)
    video_stem = Path(video_fname).stem # TODO: can cause errors if center1 and center10 both exist
    annotation_names = [f for f in fnames if (f.startswith(video_stem) and f.endswith('.h5'))]
    if len(annotation_names) > 1:
        print(f"Found more than one annotation for {video_fname}; taking first one")
    elif len(annotation_names) == 0:
        raise FileNotFoundError(f"Found no annotations for video {video_fname}")
    annotation_names = annotation_names[0]
    print(f"Using found annotations: {annotation_names}")
    return Path(annotation_dir).joinpath(annotation_names)