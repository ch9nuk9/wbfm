import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from segmentation.util.utils_metadata import DetectedNeurons

from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.external.custom_errors import ParameterTooStringentError
from wbfm.utils.projects.project_config_classes import SubfolderConfigFile
from wbfm.utils.external.utils_neuron_names import int2name_tracklet


def best_tracklet_covering_from_my_matches(df, num_frames_needed, num_frames,
                                           verbose=0):
    """
    Given a partially tracked video, choose a series of frames with enough
    tracklets, to be saved in DLC format

    Loops through windows and tracklets, to see if ALL window frames are in the tracklet
    i.e. properly rejects tracklets that skip frames
    """

    def make_window(start_frame):
        return list(range(start_frame, start_frame + num_frames_needed))

    x = list(range(num_frames - num_frames_needed))
    y = np.zeros_like(x)
    for i in x:
        which_frames = make_window(i)

        def check_for_full_covering(vals, which_frames=which_frames):
            vals = set(vals)
            return all([f in vals for f in which_frames])

        tracklets_that_cover = df['slice_ind'].apply(check_for_full_covering)
        y[i] = tracklets_that_cover.sum(axis=0)

    if len(y) == 0:
        raise ParameterTooStringentError(num_frames_needed, 'num_frames_needed')
    best_covering = np.argmax(y)
    if verbose >= 1:
        print(f"Best covering starts at volume {best_covering} with {np.max(y)} tracklets")
    if np.max(y) == 0:
        raise ParameterTooStringentError(num_frames_needed, 'num_frames_needed')

    return make_window(best_covering), y


def calculate_best_covering_from_tracklets(dlc_df: pd.DataFrame, num_training_frames: int):
    sz = dlc_df.shape
    # Only need one column from each neuron
    missing_vals = np.isnan(dlc_df.values)[:, range(0, sz[1], 4)]

    # Get number of rows without any missing values
    num_not_missing_per_window = []
    for i in tqdm(range(sz[0] - num_training_frames + 1)):
        start, stop = i, i + num_training_frames
        have_missing = np.apply_along_axis(any, 0, missing_vals[start:stop, :])
        num_not_missing_per_window.append(sum(~have_missing))

    # Get best, and convert to correct format
    y = np.array(num_not_missing_per_window)
    if len(y) == 0:
        raise ParameterTooStringentError(num_training_frames, 'num_training_frames')
    start_frame = np.argmax(y)

    best_window = list(range(start_frame, start_frame + num_training_frames))
    return best_window, y


def convert_training_dataframe_to_scalar_format(df, min_length=10, scorer=None,
                                                segmentation_metadata: DetectedNeurons = None,
                                                logger: logging.Logger = None):
    """
    Converts a dataframe of my tracklets to a format with all scalar elements

    Assumes that all neurons exist for all frames (e.g. the output from build_subset_df())

    Returns
    -------

    """

    all_dfs = []
    if segmentation_metadata is None:
        raise NotImplementedError("Must pass segmentation metadata")

    def is_valid(_df, _ind):
        _which_frames = _df.at[_ind, 'slice_ind']
        is_too_short = np.isscalar(_which_frames) or len(_which_frames) < min_length
        return not is_too_short

    logger.info("Converting to pandas multi-index format")
    for ind, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not is_valid(df, ind):
            continue

        # Get basic position and confidence data
        which_frames = df.at[ind, 'slice_ind']
        bodypart = int2name_tracklet(ind)
        zxy = row['all_xyz']
        confidence = _confidence_from_row(row, zxy)

        # Use original segmentation object to get additional data about the raw mask and brightness
        this_local_ind = row['all_ind_local']
        # NOTE: these local indices start at 0, which are direct list indices, not the dataframe indices
        this_brightness, this_volume, this_seg_id = [], [], []
        for i_local, i_frame in zip(this_local_ind, which_frames):
            this_brightness.append(segmentation_metadata.get_all_brightnesses(i_frame).iloc[i_local])
            this_volume.append(segmentation_metadata.get_all_volumes(i_frame).iloc[i_local])
            this_seg_id.append(segmentation_metadata.i_in_array_to_mask_index(i_frame, i_local))
        this_brightness = np.expand_dims(np.array(this_brightness), -1)
        this_volume = np.expand_dims(np.array(this_volume, dtype=int), -1)
        this_local_ind = np.expand_dims(np.array(this_local_ind, dtype=int), -1)
        this_seg_id = np.expand_dims(np.array(this_seg_id, dtype=int), -1)

        # Combine all
        coords = np.hstack([zxy, confidence, this_local_ind, this_seg_id, this_brightness, this_volume])

        columns = get_tracklet_dataframe_multiindex(bodypart, scorer)
        frame = pd.DataFrame(coords, columns=columns, index=which_frames)
        all_dfs.append(frame)
    if len(all_dfs) == 0:
        raise ParameterTooStringentError(min_length, 'min_length')
    new_df = pd.concat(all_dfs, axis=1)

    new_df, _ = fill_missing_indices_with_nan(new_df)

    return new_df


def _confidence_from_row(row, zxy):
    try:
        confidence = row['all_prob']
    except KeyError:
        confidence = []
    if len(confidence) == 0:
        # Then I didn't save confidences, so just set to 1
        confidence = np.ones((len(zxy), 1))
    elif len(confidence) == len(zxy) - 1:
        # The confidence usually corresponds to the match, therefore is one shorter
        confidence.append(0.0)
    return confidence


def get_tracklet_dataframe_multiindex(bodypart, scorer=None):
    column_names = tracklet_dataframe_column_names()
    if scorer is not None:
        index = pd.MultiIndex.from_product([[scorer], [bodypart], column_names],
                                           names=['scorer', 'bodyparts', 'coords'])
    else:
        index = pd.MultiIndex.from_product([[bodypart], column_names],
                                           names=['bodyparts', 'coords'])
    return index


def tracklet_dataframe_column_names():
    column_names = ['z', 'x', 'y', 'likelihood', 'raw_neuron_ind_in_list', 'raw_segmentation_id',
                    'brightness_red', 'volume']
    return column_names


def save_training_data_as_dlc_format(training_config: SubfolderConfigFile,
                                     segmentation_config: SubfolderConfigFile,
                                     DEBUG=False):
    """
    Takes my training data from my tracklet format and saves as a DLC dataframe

    Parameters
    ----------
    training_config
    DEBUG

    Returns
    -------

    """
    logging.info("Saving training data as DLC format")

    df_tracklets, df_clust, min_length_to_save, segmentation_metadata = _unpack_config_training_data_conversion(
        training_config, segmentation_config)

    # Get the frames chosen as training data, or recalculate
    num_frames = len(df_tracklets)
    which_frames = list(get_or_recalculate_which_frames(DEBUG, df_clust, num_frames, training_config))

    # Build a sub-df with only the relevant neurons; all slices
    subset_df = build_subset_df_from_tracklets(df_tracklets, which_frames)
    training_df = subset_df

    # Save
    out_fname = training_config.resolve_relative_path("training_data_tracks.h5", prepend_subfolder=True)
    training_df.to_hdf(out_fname, 'df_with_missing')

    training_config.config['df_training_3d_tracks'] = out_fname
    training_config.update_self_on_disk()

    out_fname = Path(out_fname).with_suffix(".csv")
    training_df.to_csv(str(out_fname))


def _unpack_config_training_data_conversion(training_config, segmentation_config):
    min_length_to_save = training_config.config['postprocessing_params']['min_length_to_save']
    fname = os.path.join('raw', 'clust_df_dat.pickle')
    fname = training_config.resolve_relative_path(fname, prepend_subfolder=True)
    df_clust = pd.read_pickle(fname)

    fname = training_config.resolve_relative_path_from_config('df_3d_tracklets')
    # df_tracklets: pd.DataFrame = pd.read_hdf(fname)
    df_tracklets: pd.DataFrame = pd.read_pickle(fname)

    seg_metadata_fname = segmentation_config.resolve_relative_path_from_config('output_metadata')
    segmentation_metadata = DetectedNeurons(seg_metadata_fname)

    return df_tracklets, df_clust, min_length_to_save, segmentation_metadata


def alt_save_all_tracklets_as_dlc_format(train_cfg: SubfolderConfigFile,
                                         DEBUG=False):
    """
    Takes my tracklet format and saves ALL as DLC format (i.e. many short tracklets)

    Note: uses hardcoded relative path for raw data, so it only needs to be executed in the right folder

    Parameters
    ----------
    this_config - unused
    DEBUG - unused

    Returns
    -------

    """
    logging.info("Saving all tracklets as DLC format")

    min_length = train_cfg.config['postprocessing_params']['min_length_to_save']

    raw_fname = train_cfg.resolve_relative_path(os.path.join('raw', 'clust_df_dat.pickle'), prepend_subfolder=True)
    df_raw = pd.read_pickle(raw_fname)

    df = convert_training_dataframe_to_scalar_format(df_raw, min_length=min_length, scorer=None)
    # If there are no tracklets on some frames, then there will be gaps in the indices and it will cause errors
    df, num_added = fill_missing_indices_with_nan(df)

    out_fname = train_cfg.resolve_relative_path_from_config('df_3d_tracklets')
    # out_fname = train_cfg.resolve_relative_path("all_tracklets.h5", prepend_subfolder=True)
    df.to_hdf(out_fname, 'df_with_missing')


def build_subset_df_from_tracklets(df_tracklets, which_frames, verbose=0):
    """
    Build a subset dataframe that only contains tracklets that have no nan frames for ALL of which_frames

    Parameters
    ----------
    df_tracklets
    which_frames
    verbose

    Returns
    -------

    """

    df_time_subset = df_tracklets.loc(axis=1)[:, 'z'].loc[which_frames]
    isnan_idx = df_time_subset.isna().sum() == 0
    isnan_idx = isnan_idx.droplevel(1)
    to_keep = [idx for idx in isnan_idx.index if isnan_idx[idx]]

    df_subset = df_tracklets[to_keep]
    df_subset = df_subset.reindex(columns=to_keep, level=0)  # Otherwise the dropped names remain
    return df_subset


def OLD_build_subset_df_from_tracklets(clust_df,
                                   which_frames,
                                   which_z=None,
                                   max_z_dist=1,
                                   verbose=0):
    """
    Build a dataframe that is a subset of a larger dataframe

    clust_df is my custom dataframe format

    Only keep the tracklets that pass the time and z requirements:
        - Cover each frame in which_frames
        - Not too far from which_z
    """

    ####################
    # Helper functions
    ####################

    def check_z(test_zxy, which_z=which_z, max_z_dist=max_z_dist):
        """
        Start: the centroids in all tracklets and the target z
        Return: boolean list of whether to keep or not, per tracklet
            Note that this uses the MEAN centroid, not the min or max
        """
        this_z = np.mean(np.atleast_2d(np.array(test_zxy))[:, 0])
        too_high = this_z >= (which_z + max_z_dist)
        too_low = this_z <= (which_z - max_z_dist)
        if too_high or too_low:
            to_keep = False
        else:
            to_keep = True
        return to_keep

    def check_frames(test_frames, which_frames=which_frames):
        """
        Start: the frames in all tracklets and the target frames
        Return: local indices within the test_frame to keep
            Note that this is per-tracklet
        """
        if np.isscalar(test_frames):
            return None
        test_frames_set = set(test_frames)
        local2global_ind = {}
        for f in which_frames:
            if f in test_frames_set:
                local2global_ind[test_frames.index(f)] = f
            else:
                # Must all be present
                return None
        return local2global_ind

    def keep_subset(this_ind_dict, old_ind):
        new_ind = []
        for _i in this_ind_dict:
            try:
                new_ind.append(old_ind[_i])
            except KeyError:
                continue
        return new_ind

    def rename_slices(this_ind_dict):
        return list(this_ind_dict.keys())

    ####################
    # Get indices of tracklets to keep
    ####################
    sub_df = clust_df.copy()
    # Get only the covering neurons (time)
    which_neurons = sub_df['slice_ind'].apply(check_frames)
    which_neurons_dict = which_neurons.to_dict()
    to_keep_t = [(v is not None) for k, v in which_neurons_dict.items()]
    if verbose >= 1:
        print(f"{np.count_nonzero(to_keep_t)} tracklets overlap in time")
        if verbose >= 2:
            print(to_keep_t)
    # Get close neurons (z)
    if which_z is not None:
        to_keep_z = sub_df['all_xyz'].apply(check_z)
        to_keep_z = [v for v in to_keep_z.to_dict().values()]
        to_keep = [t and z for (t, z) in zip(to_keep_t, to_keep_z)]
        if verbose >= 1:
            print(f"{np.count_nonzero(to_keep_z)} tracklets overlap in z")
            if verbose >= 2:
                print(to_keep_z)
    else:
        to_keep = to_keep_t

    for i, val in enumerate(to_keep):
        if not val:
            del which_neurons_dict[i]
    which_neurons_df = sub_df[to_keep]
    if verbose >= 1:
        print(f"Keeping {len(which_neurons_df)}/{len(clust_df)} tracklets as identifiable neurons")
        logging.debug(f"Neurons that are kept: {which_neurons_dict.keys()}")
    if len(which_neurons_df) == 0:
        # Preserve dataframe format
        return which_neurons_df

    # Get only the indices of those neurons corresponding to these frames
    names = {'all_xyz': 'all_xyz_old',
             'all_ind_local': 'all_ind_local_old',
             'all_prob': 'all_prob_old',
             'slice_ind': 'slice_ind_old'}
    out_df = which_neurons_df.rename(columns=names)

    ####################
    # Build the subset
    ####################
    # out_df['clust_ind'] = out_df['clust_ind'].astype(int)

    # All 4 fields that were renamed
    f0 = lambda df: keep_subset(which_neurons_dict[df['clust_ind']], df['all_ind_local_old'])
    out_df['all_ind_local'] = out_df.apply(f0, axis=1)

    f1 = lambda df: keep_subset(which_neurons_dict[df['clust_ind']], df['all_xyz_old'])
    out_df['all_xyz'] = out_df.apply(f1, axis=1)

    try:
        f2 = lambda df: keep_subset(which_neurons_dict[df['clust_ind']], df['all_prob_old'])
        out_df['all_prob'] = out_df.apply(f2, axis=1)
    except (IndexError, KeyError):
        # Sometimes the probability was not saved at all
        pass

    # Final one is slightly different
    # f3 = lambda df : rename_slices(which_neurons_dict[df['clust_ind']])
    f3 = lambda df: which_frames
    out_df['slice_ind'] = out_df.apply(f3, axis=1)

    return out_df


def build_subset_df_from_3dDLC(dlc3d_dlc: pd.DataFrame,
                               which_z=None,
                               max_z_dist=1,
                               verbose=0) -> pd.DataFrame:
    """
    Build a 2d DLC dataframe starting from a 3d dataframe

    Parameters
    ----------
    dlc3d_dlc

    Returns
    -------

    """

    neuron_names = get_names_from_df(dlc3d_dlc)
    names_to_keep = []

    for name in neuron_names:
        if which_z is not None:
            close_in_z = abs(which_z - np.mean(dlc3d_dlc[name]['z'])) < max_z_dist
            if close_in_z:
                names_to_keep.append(name)
        else:
            names_to_keep.append(name)

    return dlc3d_dlc[names_to_keep].copy()


def get_or_recalculate_which_frames(DEBUG, df: pd.DataFrame, num_frames: int,
                                    training_cfg: SubfolderConfigFile):
    """

    Parameters
    ----------
    DEBUG
    df - custom tracklet dataframe
    num_frames - total frames that are tracked via tracklets
    tracking_config - config file for step 3, dlc tracking

    Returns
    -------

    """
    # which_frames = this_config['track_cfg']['training_data_3d']['which_frames']
    which_frames = training_cfg.config['training_data_3d'].get('which_frames', None)

    if which_frames is None:
        # Choose a subset of frames with enough tracklets
        num_frames_needed = training_cfg.config['training_data_3d']['num_training_frames']
        tracklet_opt = {'num_frames_needed': num_frames_needed,
                        'num_frames': num_frames,
                        'verbose': 1}
        if DEBUG:
            tracklet_opt['num_frames_needed'] = 2
        which_frames, _ = best_tracklet_covering_from_my_matches(df, **tracklet_opt)

        training_cfg.config['training_data_3d']['which_frames'] = which_frames
        training_cfg.update_self_on_disk()

    return which_frames


def build_dlc_annotation_one_tracklet(row,
                                      bodypart,
                                      num_frames=1000,
                                      coord_names=None,
                                      which_frame_subset=None,
                                      scorer=None,
                                      min_length=5,
                                      neuron_ind=1,
                                      relative_imagenames=None,
                                      verbose=0):
    """
    Builds DLC-style dataframe and .h5 annotation from my tracklet dataframe

    Can also be 3d if coord_names is passed as ['z', 'x', 'y', 'likelihood']
    """
    if coord_names is None:
        coord_names = ['x', 'y', 'likelihood']

    # Variables to be written
    if scorer is None:
        scorer = 'feature_tracker'
    if relative_imagenames is None:
        # Just frame number
        index = list(range(num_frames))
    else:
        index = relative_imagenames

    tracklet_length = len(row['all_xyz'])

    if verbose >= 2:
        print(f"Found tracklet of length {tracklet_length}")
    if tracklet_length < min_length:
        return None

    # Relies on ZXY format for this_xyz column in the original dataframe
    coord_mapping = {'z': 0, 'x': 1, 'y': 2}

    # Build a dataframe for one neuron across all frames
    # Will be zeros if not detected in a given frame
    coords = np.zeros((num_frames, len(coord_names),))
    # This should zip through all_xyz, but all_prob might be empty
    slice_ind, all_xyz, all_prob = row['slice_ind'], row['all_xyz'], row['all_prob']
    if len(all_prob) < len(all_xyz):
        all_prob = [1.0 for _ in all_xyz]

    for this_slice, this_xyz, this_prob in zip(slice_ind, all_xyz, all_prob):
        # this_xyz is format ZXY
        for i, coord_name in enumerate(coord_names):
            if coord_name in coord_mapping:
                # is spatial
                coords[this_slice, i] = int(this_xyz[coord_mapping[coord_name]])
            else:
                # is non-spatial, i.e. likelihood
                try:
                    coords[this_slice, -1] = this_prob
                except IndexError:
                    coords[this_slice, -1] = 0.0
                    pass
    if which_frame_subset is not None:
        # error
        coords = coords[which_frame_subset, :]

    m_index = pd.MultiIndex.from_product([[scorer], [bodypart],
                                          coord_names],
                                         names=['scorer', 'bodyparts', 'coords'])
    frame = pd.DataFrame(coords, columns=m_index, index=index)

    return frame


def build_dlc_annotation_from_tracklets(clust_df: pd.DataFrame,
                                        min_length: int,
                                        num_frames: int = 1000,
                                        coord_names: list = None,
                                        scorer: str = None,
                                        relative_imagenames: list = None,
                                        which_frame_subset: list = None,
                                        verbose: int = 1) -> pd.DataFrame:
    """
    Builds a 2d DLC style annotation using tracklet dataframe (my custom format)

    Parameters
    ----------
    clust_df
    min_length
    num_frames
    coord_names
    scorer
    relative_imagenames
    which_frame_subset
    verbose

    Returns
    -------

    """
    new_dlc_df = None
    if coord_names is None:
        coord_names = ['x', 'y', 'likelihood']
    # all_bodyparts = np.asarray(clust_df['clust_ind'])

    neuron_ind = 1
    options = {'min_length': min_length,
               'num_frames': num_frames,
               'coord_names': coord_names,
               'relative_imagenames': relative_imagenames,
               'which_frame_subset': which_frame_subset,
               'scorer': scorer,
               'verbose': verbose - 1}
    for i, row in tqdm(clust_df.iterrows(), total=clust_df.shape[0]):
        options['neuron_ind'] = neuron_ind
        ind = row['clust_ind']
        bodypart = f'neuron{ind}'
        frame = build_dlc_annotation_one_tracklet(row, bodypart, **options)
        if frame is not None:
            new_dlc_df = pd.concat([new_dlc_df, frame], axis=1)
            neuron_ind = neuron_ind + 1
    if verbose >= 1 and new_dlc_df is not None:
        print(f"Found {len(new_dlc_df.columns) / len(coord_names)} tracks of length >{min_length}")

    return new_dlc_df


def build_dlc_annotation_from_3dDLC(subset_df: pd.DataFrame,
                                    min_length: int,
                                    num_frames: int = 1000,
                                    coord_names: list = None,
                                    scorer: str = None,
                                    relative_imagenames: list = None,
                                    which_frame_subset: list = None,
                                    verbose: int = 1) -> pd.DataFrame:
    """
    Builds a 2d DLC style annotation using tracklet dataframe (my custom format)

    Parameters
    ----------
    subset_df
    min_length
    num_frames
    coord_names
    scorer
    relative_imagenames
    which_frame_subset
    verbose

    Returns
    -------

    """
    new_dlc_df = None
    if coord_names is None:
        coord_names = ['x', 'y', 'likelihood']

    neuron_ind = 1
    options = {'min_length': min_length,
               'num_frames': num_frames,
               'coord_names': coord_names,
               'relative_imagenames': relative_imagenames,
               'which_frame_subset': which_frame_subset,
               'scorer': scorer,
               'verbose': verbose - 1}

    neuron_names = get_names_from_df(subset_df)
    neuron_names = [n for n in neuron_names if n in subset_df]

    for name in tqdm(neuron_names):
        bodypart = name  # Keep original neuron name
        frame = build_dlc_annotation_one_dlc3d_subset(subset_df[name], bodypart, **options)
        if frame is not None:
            new_dlc_df = pd.concat([new_dlc_df, frame], axis=1)
            neuron_ind = neuron_ind + 1
    if verbose >= 1 and new_dlc_df is not None:
        print(f"Found {len(new_dlc_df.columns) / len(coord_names)} tracks of length >{min_length}")

    # print("New dataframe written:")
    # print(new_dlc_df)

    return new_dlc_df


def build_dlc_annotation_one_dlc3d_subset(row,
                                          bodypart,
                                          num_frames=1000,
                                          coord_names=None,
                                          which_frame_subset=None,
                                          scorer=None,
                                          min_length=5,
                                          relative_imagenames=None,
                                          verbose=0):
    """
    Builds DLC-style dataframe and .h5 annotation from my tracklet dataframe

    Can also be 3d if coord_names is passed as ['z', 'x', 'y', 'likelihood']
    """
    if coord_names is None:
        coord_names = ['x', 'y', 'likelihood']

    # Variables to be written
    if scorer is None:
        scorer = 'feature_tracker'
    if relative_imagenames is None:
        # Just frame number
        index = list(range(num_frames))
    else:
        index = relative_imagenames

    # tracklet_length = len(row['all_xyz'])
    #
    # if verbose >= 2:
    #     print(f"Found tracklet of length {tracklet_length}")
    # if tracklet_length < min_length:
    #     return None
    # Relies on ZXY format for this_xyz column in the original dataframe
    # coord_mapping = {'z': 0, 'x': 1, 'y': 2}
    # Build a dataframe for one neuron across all frames
    # Will be zeros if not detected in a given frame
    # coords = np.zeros((num_frames, len(coord_names), ))
    # This should zip through all_xyz, but all_prob might be empty
    # slice_ind, all_xyz, all_prob = row['slice_ind'], row['all_xyz'], row['all_prob']
    # if len(all_prob) < len(all_xyz):
    #     all_prob = [1.0 for _ in all_xyz]
    # for this_slice, this_xyz, this_prob in zip(slice_ind, all_xyz, all_prob):
    #     # this_xyz is format ZXY
    #     for i, coord_name in enumerate(coord_names):
    #         if coord_name in coord_mapping:
    #             # is spatial
    #             coords[this_slice, i] = int(this_xyz[coord_mapping[coord_name]])
    #         else:
    #             # is non-spatial, i.e. likelihood
    #             try:
    #                 coords[this_slice, -1] = this_prob
    #             except:
    #                 coords[this_slice, -1] = 0.0
    #                 pass
    # if which_frame_subset is not None:
    #     # error
    #     coords = coords[which_frame_subset, :]

    coords = row[coord_names].to_numpy()

    m_index = pd.MultiIndex.from_product([[scorer], [bodypart],
                                          coord_names],
                                         names=['scorer', 'bodyparts', 'coords'])
    frame = pd.DataFrame(coords, columns=m_index, index=index)

    return frame


def modify_config_files_for_training_data(project_config, segment_cfg, training_cfg):
    # Modify the config files so that we process the training data instead of the main masks
    segment_cfg.config['output_masks'] = training_cfg.config['reindexed_masks']
    segment_cfg.config['output_metadata'] = training_cfg.config['reindexed_metadata']
    project_config.config['dataset_params']['num_frames'] = training_cfg.config['training_data_3d'][
        'num_training_frames']
    start_volume = training_cfg.config['training_data_3d']['which_frames'][0]
    project_config.config['deprecated_dataset_params']['start_volume'] = start_volume


def translate_training_names_to_raw_names(df_training_data):
    """As of 1/24/2022, the columns should have the SAME names"""
    return get_names_from_df(df_training_data)
    # offset_names = list(df_training_data.columns.levels[0])
    # ind = [name2int_neuron_and_tracklet(n) for n in offset_names]
    # training_tracklet_names = [int2name_tracklet(i - 1) for i in ind]
    # return training_tracklet_names
