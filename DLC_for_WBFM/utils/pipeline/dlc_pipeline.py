from DLC_for_WBFM.utils.preprocessing.convert_matlab_annotations_to_DLC import csv_annotations2config_names
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings, perform_preprocessing
from DLC_for_WBFM.utils.video_and_data_conversion.video_conversion_utils import write_numpy_as_avi
from DLC_for_WBFM.utils.feature_detection.utils_networkx import calc_bipartite_from_distance
# from DLC_for_WBFM.utils.feature_detection.visualize_using_dlc import build_subset_df, build_dlc_annotation_all, build_relative_imagenames, save_dlc_annotations
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import best_tracklet_covering
from DLC_for_WBFM.utils.video_and_data_conversion.import_video_as_array import get_single_volume
from DLC_for_WBFM.utils.preprocessing.DLC_utils import get_annotations_from_dlc_config, get_z_from_dlc_name, update_pose_config, training_data_from_annotations, build_png_training_data, create_dlc_project
import pandas as pd
from DLC_for_WBFM.utils.feature_detection.visualization_tracks import visualize_tracks
import numpy as np
import tifffile
import pickle
from pathlib import Path
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
    # OPTIMIZE: for now, requires re-preprocessing

    def make_avi_name(center):
        fname = f"center{center}.avi"  # NOT >8 CHAR (without .avi)
        if len(fname) > 12:
            # BUG: fix required short filenames
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
        dlc_cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_config)
        video_list = list(dlc_cfg['video_sets'].keys())
        # Works even if already analyzed
        deeplabcut.analyze_videos(dlc_config, video_list)
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


def get_traces_from_3d_tracks(segment_cfg,
                              track_cfg,
                              traces_cfg,
                              dataset_params,
                              DEBUG=False):
    """
    Connect the 3d traces to previously segmented masks

    Get both red and green traces for each neuron
    """
    # Settings
    max_dist = track_cfg['final_3d_tracks']['max_dist_to_segmentation']
    start_volume = dataset_params['start_volume']
    num_frames = dataset_params['num_frames']
    # Get previous annotations
    segmentation_fname = segment_cfg['output']['metadata']
    with open(segmentation_fname, 'rb') as f:
        segmentation_metadata = pickle.load(f)
    dlc_fname = track_cfg['final_3d_tracks']['df_fname']
    dlc_tracks = pd.read_hdf(dlc_fname)

    # Convert DLC dataframe to array
    all_neuron_names = list(dlc_tracks.columns.levels[0])

    def get_dlc_zxy(t, dlc_tracks=dlc_tracks):
        all_dlc_zxy = np.zeros((len(all_neuron_names), 3))
        coords = ['z', 'y', 'x']
        for i, name in enumerate(all_neuron_names):
            all_dlc_zxy[i, :] = np.asarray(dlc_tracks[name][coords].loc[t])
        return all_dlc_zxy

    # Main loop: Match segmentations to tracks
    # Also: get connected red brightness and mask

    # Initialize multi-index dataframe for data
    frame_list = list(range(start_volume, num_frames + start_volume))
    # red_brightness = {}  # key = neuron id (int); val = list
    # red_
    save_names = ['brightness', 'volume', 'centroid_ind', 'z', 'x', 'y']
    m_index = pd.MultiIndex.from_product([all_neuron_names,
                                         save_names],
                                         names=['neurons', 'data'])
    sz = (len(frame_list), len(m_index))
    empty_dat = np.empty(sz)
    empty_dat[:] = np.nan
    red_dat = pd.DataFrame(empty_dat,
                           columns=m_index,
                           index=frame_list)

    all_matches = {}  # key = i_vol; val = 3xN-element list
    print("Looping through frames:")
    for i_volume in tqdm(frame_list):
        # Get DLC point cloud
        # NOTE: This dataframe starts at 0, not start_volume
        zxy0 = get_dlc_zxy(i_volume)
        # REVIEW: Get segmentation point cloud
        seg_zxy = segmentation_metadata[i_volume]['centroids']
        seg_zxy = [np.asarray(row) for row in seg_zxy]
        zxy1 = np.array(seg_zxy)
        # Get matches
        out = calc_bipartite_from_distance(zxy0, zxy1, max_dist=max_dist)
        matches, conf, _ = out
        if DEBUG:
            visualize_tracks(zxy0, zxy1, matches)
        # Use metadata to get red traces
        # OPTIMIZE: minimum confidence?
        mdat = segmentation_metadata[i_volume]
        all_seg_names = list(mdat['centroids'].keys())
        # TODO: is this actually setting?
        for i_dlc, i_seg in matches:
            d_name = all_neuron_names[i_dlc]  # output name
            s_name = all_seg_names[i_seg]
            # See saved_names above
            i = i_volume
            red_dat[(d_name, 'brightness')].loc[i] = mdat['total_brightness'][s_name]
            red_dat[(d_name, 'volume')].loc[i] = mdat['neuron_volume'][s_name]
            red_dat[(d_name, 'z')].loc[i] = mdat['centroids'][s_name][0]
            red_dat[(d_name, 'x')].loc[i] = mdat['centroids'][s_name][1]
            red_dat[(d_name, 'y')].loc[i] = mdat['centroids'][s_name][2]
            # print(red_dat[d_name, :])

        # Save
        all_matches[i_volume] = list(zip(matches, conf))

    # TODO: Get full green traces using masks

    # Save traces (red and green) and neuron names
    fname = Path('4-traces').joinpath('red_traces.h5')
    red_dat.to_hdf(fname, "df_with_missing")
    traces_cfg['traces']['red'] = str(fname)

    traces_cfg['traces']['neuron_names'] = all_neuron_names

    # Also save matches as a separate file
    # ENHANCE: save as part of the dataframes?
    fname = Path('4-traces').joinpath('all_matches.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(all_matches, f)
    traces_cfg['all_matches'] = str(fname)

    # Save the output filenames
    edit_config(traces_cfg['self_path'], traces_cfg)
