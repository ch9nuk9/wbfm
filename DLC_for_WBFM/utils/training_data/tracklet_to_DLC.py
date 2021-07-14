import os
from pathlib import Path

import numpy as np
import pandas as pd


def best_tracklet_covering(df, num_frames_needed, num_frames,
                           verbose=0):
    """
    Given a partially tracked video, choose a series of frames with enough
    tracklets, to be saved in DLC format

    Loops through windows and tracklets, to see if ALL window frames are in the tracklet
    i.e. properly rejects tracklets that skip frames
    """

    def make_window(start_frame):
        return list(range(start_frame,start_frame+num_frames_needed+1))

    x = list(range(num_frames-num_frames_needed))
    y = np.zeros_like(x)
    for i in x:
        which_frames = make_window(i)
        def check_for_full_covering(vals, which_frames=which_frames):
            vals = set(vals)
            return all([f in vals for f in which_frames])

        tracklets_that_cover = df['slice_ind'].apply(check_for_full_covering)
        y[i] = tracklets_that_cover.sum(axis=0)

    best_covering = np.argmax(y)
    if verbose >= 1:
        print(f"Best covering starts at volume {best_covering} with {np.max(y)} tracklets")

    return make_window(best_covering), y


def convert_training_dataframe_to_dlc_format(df, scorer='Charlie'):
    """
    Converts a dataframe of my tracklets to DLC format

    Assumes that all neurons exist for all frames (e.g. the output from build_subset_df())

    Returns
    -------

    """

    ind = list(df.index)[0]
    which_frames = df.at[ind, 'slice_ind']
    new_df = None

    for ind, row in df.iterrows():
        bodypart = f'neuron{ind}'
        confidence = row['all_prob']
        zxy = row['all_xyz']
        if len(confidence) == 0:
            # Then I didn't save confidences, so just set to 1
            confidence = np.ones((len(zxy), 1))
        coords = np.hstack([zxy, confidence])

        index = pd.MultiIndex.from_product([[scorer], [bodypart], ['z', 'x', 'y', 'likelihood']],
                                           names=['scorer', 'bodyparts', 'coords'])
        frame = pd.DataFrame(coords, columns=index, index=which_frames)
        new_df = pd.concat([new_df, frame], axis=1)

    return new_df


def save_training_data_as_dlc_format(this_config, DEBUG=False):
    """
    Takes my tracklet format and saves as a DLC dataframe

    Parameters
    ----------
    this_config
    DEBUG

    Returns
    -------

    """

    fname = os.path.join('2-training_data', 'raw', 'clust_df_dat.pickle')
    df = pd.read_pickle(fname)

    # Get the frames chosen as training data, or recalculate
    which_frames = list(get_or_recalculate_which_frames(DEBUG, df, this_config))

    # Build a sub-df with only the relevant neurons; all slices
    # Todo: connect up to actually tracked z slices?
    subset_opt = {'which_z': None,
                  'max_z_dist': None,
                  'verbose': 1}
    subset_df = build_subset_df(df, which_frames, **subset_opt)
    training_df = convert_training_dataframe_to_dlc_format(subset_df, scorer='Charlie')

    out_fname = os.path.join("2-training_data", "training_data_tracks.h5")
    training_df.to_hdf(out_fname, 'df_with_missing')

    out_fname = Path(out_fname).with_suffix(".csv")
    training_df.to_csv(out_fname)


def build_subset_df(clust_df,
                    which_frames,
                    which_z=None,
                    max_z_dist=1,
                    verbose=0):
    """
    Build a dataframe that is a subset of a larger dataframe

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
        too_high = this_z >= (which_z+max_z_dist)
        too_low = this_z <= (which_z-max_z_dist)
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
        for i in this_ind_dict:
            try:
                new_ind.append(old_ind[i])
            except:
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
        print(f"Keeping {len(which_neurons_df)}/{len(clust_df)} tracklets")
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
    # All 4 fields that were renamed
    f0 = lambda df: keep_subset(which_neurons_dict[df['clust_ind']], df['all_ind_local_old'])
    out_df['all_ind_local'] = out_df.apply(f0, axis=1)

    f1 = lambda df: keep_subset(which_neurons_dict[df['clust_ind']], df['all_xyz_old'])
    out_df['all_xyz'] = out_df.apply(f1, axis=1)

    f2 = lambda df: keep_subset(which_neurons_dict[df['clust_ind']], df['all_prob_old'])
    out_df['all_prob'] = out_df.apply(f2, axis=1)

    # Final one is slightly different
    # f3 = lambda df : rename_slices(which_neurons_dict[df['clust_ind']])
    f3 = lambda df: which_frames
    out_df['slice_ind'] = out_df.apply(f3, axis=1)

    return out_df


def get_or_recalculate_which_frames(DEBUG, df, this_config):
    try:
        which_frames = this_config['track_cfg']['training_data_3d']['which_frames']
    except KeyError:
        which_frames = None
    if which_frames is None:
        # Choose a subset of frames with enough tracklets
        num_frames_needed = this_config['track_cfg']['training_data_3d']['num_training_frames']
        tracklet_opt = {'num_frames_needed': num_frames_needed,
                        'num_frames': this_config['dataset_params']['num_frames'],
                        'verbose': 1}
        if DEBUG:
            tracklet_opt['num_frames_needed'] = 2
        which_frames, _ = best_tracklet_covering(df, **tracklet_opt)
    return which_frames