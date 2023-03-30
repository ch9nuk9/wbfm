import copy
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Set
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df, get_next_name_generator, \
    split_single_sparse_tracklet, split_multiple_tracklets
from wbfm.utils.general.custom_errors import NoMatchesError, AnalysisOutOfOrderError
from wbfm.utils.projects.utils_neuron_names import name2int_neuron_and_tracklet, \
    int2name_using_mode


def create_new_track(i0, i1,
                     i0_xyz,
                     i1_xyz,
                     i1_prob,
                     next_clust_ind,
                     this_point_cloud_offset,
                     next_point_cloud_offset,
                     which_slice,
                     i1_global,
                     verbose=0):
    new_track = pd.DataFrame({'clust_ind': next_clust_ind,
                              'all_ind_local': [[i0, i1]],
                              'all_ind_global': [[this_point_cloud_offset + i0, i1_global]],
                              'all_xyz': [[i0_xyz, i1_xyz]],
                              'all_prob': [[i1_prob]],
                              'slice_ind': [[which_slice, which_slice + 1]],
                              'extended_this_slice': True,
                              'not_finished': True})

    if verbose >= 1:
        print(f"Creating new track {next_clust_ind}")
    next_clust_ind = next_clust_ind + 1

    return next_clust_ind, new_track


def append_to_track(row, i, clust_df, which_slice, i1, i1_global,
                    i1_xyz, i1_prob,
                    append_or_extend=list.append,
                    verbose=0):
    """
    Parameters
    ===============
    row : dataframe
        As the full dataframe is iterated through, this is the row to be extended
    i : int
        The index of the full dataframe row that is being appended
    clust_df : dataframe
        The full dataframe. The 'i'th row should be 'row' (above)
    which_slice : int
        The slice (frame) corresponding to the index 'i'
    i1 : int
        The local index of the matched neuron
    i1_global : int
        The global coordinate representation of 'i1'
    i1_xyz : tuple(x,y,z)
        The xyz coordinates of the neuron 'i1'
    i1_prob : float
        The probability of the neuron being added to this tracklet
        For now, determined by simple sequential matching
    append_or_extend : function
        list.append or list.extend
        Usage:
            list.append is used when adding a single neuron to the row
            list.extend allows full rows to be combined

    """
    r = row['all_ind_local'][:]
    append_or_extend(r, i1)
    clust_df.at[i, 'all_ind_local'] = r

    r = row['all_ind_global'][:]
    append_or_extend(r, i1_global)
    clust_df.at[i, 'all_ind_global'] = r

    r = row['all_xyz'][:]
    if type(i1_xyz) is np.ndarray:
        # row['all_xyz'] = np.vstack([row['all_xyz'], i1_xyz])
        clust_df.at[i, 'all_xyz'] = np.vstack([r, i1_xyz])
    else:
        append_or_extend(r, i1_xyz)
        clust_df.at[i, 'all_xyz'] = r
    append_or_extend(row['all_prob'], i1_prob)

    r = row['all_prob'][:]
    clust_df.at[i, 'all_prob'] = r

    r = row['slice_ind'][:]
    if type(which_slice) is int:
        append_or_extend(r, which_slice + 1)
    else:
        # Assume they are already the to-be-matched indices
        append_or_extend(r, which_slice)
    clust_df.at[i, 'slice_ind'] = r

    clust_df.at[i, 'extended_this_slice'] = True

    if verbose >= 1:
        clust_ind = row['clust_ind']
        tmp = len(row['all_ind_local'])
        print(f"Adding to track {clust_ind} (length {tmp}) on slice {which_slice}")

    return clust_df


##
## Main function
##

def build_tracklets_from_matches(all_neurons,
                                 all_matches,
                                 all_likelihoods=None,
                                 verbose=0):
    """
    Builds tracklets from an array of pairwise matches

    Parameters
    ===============
    all_neurons : list
        List of lists, with integers for all neurons present in a frame
    all_matches : list
        Matches between adjacent frames in the coordinates local to each frame
    all_likelihoods : list
        Same as all_matches, but for the confidence of the matchs

    See build_tracklets_from_classes() if using ReferenceFrame class
    """

    from wbfm.utils.neuron_matching.class_reference_frame import ReferenceFrame
    if type(all_neurons[0]) == ReferenceFrame:
        print("===============")
        print("Found type ReferenceFrame")
        print("Did you mean to call build_tracklets_from_classes()?")
        raise ValueError

    if all_likelihoods is None:
        all_likelihoods = [np.ones(len(m)) for m in all_matches]

    # Use registration results to build a combined and colored pointcloud
    columns = ['clust_ind', 'all_ind_local', 'all_ind_global', 'all_xyz',
               'all_prob',
               'slice_ind', 'extended_this_slice', 'not_finished']
    clust_df = pd.DataFrame(columns=columns)

    next_clust_ind = 0
    this_point_cloud_offset = 0
    next_point_cloud_offset = 0

    # for i_match, match in tqdm(enumerate(all_matches), total=len(all_matches)):
    for i_match, match in enumerate(all_matches):
        if verbose >= 2:
            print("==============================================================")
            print(f"{i_match} / {len(all_matches)}")
            print(f"Found {len(match)} matches")
        # Get transform to global coordinates
        these_neurons = all_neurons[i_match]
        this_xyz = np.asarray(these_neurons)
        next_neurons = all_neurons[i_match + 1]
        next_xyz = np.asarray(next_neurons)
        next_point_cloud_offset = next_point_cloud_offset + len(these_neurons)

        # Get the probabilities as well
        this_prob = all_likelihoods[i_match]
        this_prob = np.asarray(this_prob)
        # next_prob = all_likelihoods[i_match+1]
        # next_prob = np.asarray(next_prob)

        offsets = {'next_point_cloud_offset': next_point_cloud_offset,
                   'this_point_cloud_offset': this_point_cloud_offset,
                   'which_slice': i_match}

        pairs = np.asarray(match)
        # Initialize ALL as to-be-finished
        clust_df['extended_this_slice'] = False

        all_new_tracks = []
        for i_pair, (i0, i1) in enumerate(pairs):
            next_point = {'i1': i1,
                          'i1_xyz': next_xyz[i1, :],
                          'i1_prob': this_prob[i_pair],  # Probability is attached to the pair
                          'i1_global': next_point_cloud_offset + i1}
            current_point = {'i0': i0,
                             'i0_xyz': this_xyz[i0, :],
                             'next_clust_ind': next_clust_ind}
            # If no tracks, initialize
            ind_to_check = clust_df['not_finished'] & ~clust_df['extended_this_slice']

            if verbose >= 2:
                print(f"Clusters available to check: {np.where(ind_to_check)}")
            if len(ind_to_check) == 0:
                next_clust_ind, new_track = create_new_track(**current_point, **next_point, **offsets)
                all_new_tracks.append(new_track)
                continue
            # Add to previous track if possible
            for i, row in clust_df[ind_to_check].iterrows():
                if verbose >= 2:
                    print(f"pair: {i0}, {i1}, trying cluster: {i}")
                if i0 == row['all_ind_local'][-1]:
                    clust_df = append_to_track(row, i, clust_df, i_match, **next_point)
                    break
            else:
                # Create new track
                next_clust_ind, new_track = create_new_track(**current_point, **next_point, **offsets)
                all_new_tracks.append(new_track)

        # Actually add the tracks to the dataframe
        for t in all_new_tracks:
            clust_df = clust_df.append(t, ignore_index=True)

        # Finalize tracks that didn't get a new point this loop
        to_finish = ~clust_df['extended_this_slice'].astype(bool)
        if len(np.where(to_finish)[0]) > 0 and verbose >= 1:
            print(f"Finished tracks {np.where(to_finish)[0]}")
        clust_df.loc[to_finish, 'not_finished'] = False

        if verbose >= 3 and len(pairs) > 0:
            print("WIP")
            # visualize_tracks_simple(this_pc, next_pc, pairs)

        this_point_cloud_offset = next_point_cloud_offset

    clust_df['all_xyz'] = clust_df['all_xyz'].apply(np.array)

    return clust_df


def build_tracklets_from_classes(all_frames,
                                 all_matches_dict,
                                 all_likelihoods_dict=None,
                                 verbose=0):
    """
    Build tracklets starting from a different format

    Parameters
    ===========================
    all_frames - list
        Simple list of ReferenceFrame objects
    all_matches_dict - dict
        Dictionary of matches, which are lists of local neuron indices
        all_matches_dict[(0,1)] = list(list())
    all_likelihoods_dict - dict
        Same format as all_matches_dict

    See also: build_tracklets_from_matches
    """

    if type(all_matches_dict) != dict:
        print("Expected dictionary of pairwise matches")
        print("Did you mean to call build_tracklets_from_matches()?")
        raise ValueError

    # Input of build_tracklets_from_matches:
    #   1. List of all pairwise matches
    #   2. List of all neuron 3d locations
    # if type(all_frames)==dict:
    # BUG: make the below loops work for dict
    # all_frames = list(all_frames.values())
    # print("If this is a dict, then the indices are probably off.")
    # raise ValueError
    try:
        all_neurons = [all_frames[0].neuron_locs]
        final_frame_ind = len(all_frames)
        start_frame_ind = 0
    except KeyError:
        k = list(all_frames)
        all_neurons = [all_frames[k[0]].neuron_locs]
        start_frame_ind = min(k)
        final_frame_ind = max(k)
    all_matches = []
    if all_likelihoods_dict is None:
        all_likelihoods = None
    else:
        all_likelihoods = []

    if verbose > 1:
        print("Casting class data in list form...")
    nonzero_matches = 0
    for i in range(1, final_frame_ind):
        # Pad the initials with empties if this is a dict
        # for key, i in zip(all_matches_dict, all_frames):
        # Get matches and conf
        key = (i - 1, i)
        if key in all_matches_dict:
            all_matches.append(all_matches_dict[key])
            nonzero_matches += 1
        else:
            all_matches.append([])
        if all_likelihoods is not None:
            all_likelihoods.append(all_likelihoods_dict[key])

        if i < start_frame_ind:
            all_neurons.append([])
        else:
            all_neurons.append(all_frames[i].neuron_locs)
        # all_neurons.append(frame.neuron_locs)
    if verbose > 1:
        print(f"Found {nonzero_matches} nonzero matches")
    if nonzero_matches == 0:
        print("Found no matches; is the dictionary in the proper format?")
        return None

    # Call old function
    if verbose > 1:
        print("Calling build_tracklets_from_matches()")
    return build_tracklets_from_matches(all_neurons,
                                        all_matches,
                                        all_likelihoods,
                                        verbose=verbose)


def build_tracklets_simple(all_matches, verbose=0):
    """
    Build tracklets without requiring other data
        Especially neuron position

    See also: build_tracklets_from_matches
    """

    num_neurons = 0
    for m in all_matches:
        max_neuron = np.max(np.array(m))
        num_neurons = max(num_neurons, max_neuron)

    def dummy_xyz_factory(num_neurons=num_neurons):
        return np.zeros((num_neurons, 3))

    all_neurons = defaultdict(dummy_xyz_factory)

    # Call old function
    return build_tracklets_from_matches(all_neurons,
                                        all_matches,
                                        None,
                                        verbose=verbose)


##
## Postprocessing: stitching tracklets together
##

def consolidate_tracklets(df_raw, tracklet_matches, verbose=0):
    """Consolidate tracklets using matches

    Note: assumes that the indices in tracklet_matches correspond to the clust_ind column
    """
    base_of_dropped_rows = {}
    rows_to_drop = set()
    df = copy.deepcopy(df_raw)
    for row0_ind, row1_ind in tracklet_matches:
        # If we have two matches: (0,1) and later (1,10), add directly to track 0
        # BUG: what if the matches are out of order?
        if row0_ind in rows_to_drop:
            row0_ind = base_of_dropped_rows[row0_ind]
        base_row = df.loc[row0_ind].copy(deep=True)
        row_to_add = df.loc[row1_ind].copy(deep=True)
        if verbose >= 2:
            print(f"Adding track {row1_ind} to track {row0_ind}")

        df = append_to_track(base_row,
                             row0_ind,
                             df,
                             row_to_add['slice_ind'][:],
                             row_to_add['all_ind_local'][:],
                             row_to_add['all_ind_global'][:],
                             row_to_add['all_xyz'][:],
                             row_to_add['all_prob'][:],
                             append_or_extend=list.extend)
        base_of_dropped_rows[row1_ind] = row0_ind
        rows_to_drop.add(row1_ind)

    if verbose >= 1:
        print(f"Extended and dropped {len(rows_to_drop)}/{df.shape[0]} rows")

    return df.drop(rows_to_drop, axis=0)


##
## API functions
##

def convert_from_dict_to_lists(tmp_matches, tmp_conf, tmp_neurons):
    # Convert from dict and Frame objects to just lists
    all_matches, all_conf = [], []
    all_neurons = []
    for i in range(len(tmp_matches) - 1):
        # Assume keys describe pairwise matches
        k = (i, i + 1)
        all_matches.append(tmp_matches[k])
        all_conf.append(tmp_conf[k])
        # This is a ReferenceFrame object
        all_neurons.append(tmp_neurons[i].neuron_locs)
    # This list is one element longer
    all_neurons.append(tmp_neurons[-1].neuron_locs)

    return all_matches, all_conf, all_neurons


##
## Massive simplification / refactor
##

def build_tracklets_dfs(pairwise_matches_dict: dict,
                        zxy_per_neuron_per_frame: dict = None,
                        slice_offset: int = 0,
                        verbose=0) -> pd.DataFrame:
    """
    Instead of looping through pairs, does a depth-first-search to fully complete a tracklet, then moves to the next

    Expects DICT for pairwise_matches_dict
    """
    if len(pairwise_matches_dict) == 0:
        raise NoMatchesError("No matches found")
    assert len(list(pairwise_matches_dict.keys())[0]) == 2, "Dictionary should be indexed by a tuple"
    assert len(
        list(pairwise_matches_dict.values())) >= 2, "Dictionary should contain match indices, confidence optional"

    # Make everything a dictionary
    dict_of_match_dicts = {k: dict([m0[:2] for m0 in m]) for k, m in pairwise_matches_dict.items()}
    try:
        dict_of_prob_dicts = {k: {m0[0]: m0[2] for m0 in m} for k, m in pairwise_matches_dict.items()}
        has_probability = True
    except IndexError:
        has_probability = False
    # f = lambda m: len(m)
    list_of_all_possible_matches = list(map(len, pairwise_matches_dict.values()))
    total_possible_matches = sum(list_of_all_possible_matches)
    # mean_progress_per_match = int(np.mean(np.array(list_of_all_possible_matches)))

    min_pair = min([k[0] for k in pairwise_matches_dict.keys()])
    max_pair = max([k[0] for k in pairwise_matches_dict.keys()])
    tracklet_starting_indices = list(range(min_pair, max_pair))

    def get_start_match(match_dicts):
        # Note: match_dicts will progressively have entries deleted
        for i in tracklet_starting_indices:
            # Make sure order is respected
            match_key = (i, i + 1)
            match_dict = match_dicts.get(match_key, [])
            if len(match_dict) == 0:
                continue
            for k, v in match_dict.items():
                # Order doesn't matter in this dict
                return match_key, k, v
        return None, None, None

    # List of dataframes that will be concatenated
    all_dfs = []

    # Individual tracks
    clust_ind = 0
    if verbose >= 1:
        print(f"Searching through total possible matches: {total_possible_matches}")
    with tqdm(total=total_possible_matches, leave=False) as pbar:
        while True:
            # Choose a starting point, and initialize lists
            match_key, i0, i1 = get_start_match(dict_of_match_dicts)
            if match_key is None:
                if verbose >= 1:
                    print(f"No matches left")
                break
            if verbose >= 2:
                print(f"Starting match: {match_key}, {i0}, {i1}")
            i_frame0, i_frame1 = match_key

            all_ind_local = [i0, i1]
            if zxy_per_neuron_per_frame is not None:
                all_zxy = [zxy_per_neuron_per_frame[i_frame0][i0], zxy_per_neuron_per_frame[i_frame1][i1]]
            else:
                all_zxy = [[], []]
            slice_ind = [i_frame0, i_frame1]
            if has_probability:
                all_prob = [dict_of_prob_dicts[match_key][i0]]
            else:
                all_prob = []

            # Remove match
            del dict_of_match_dicts[match_key][i0]

            # DFS for this starting point
            remaining_frame_pair_indices = list(range(match_key[1], tracklet_starting_indices[-1]+1))
            if verbose >= 3:
                # Note: even
                print(f"Searching through {len(remaining_frame_pair_indices)} remaining pairs")
            for i_pair in remaining_frame_pair_indices:
                next_match_key = (i_pair, i_pair + 1)
                next_match_dict = dict_of_match_dicts.get(next_match_key, {})
                if i1 in next_match_dict:
                    # These lines are just saving this match
                    i0, i1 = i1, next_match_dict[i1]
                    i_frame = next_match_key[1]

                    all_ind_local.append(i1)
                    if zxy_per_neuron_per_frame is not None:
                        all_zxy.append(zxy_per_neuron_per_frame[i_frame][i1])
                    else:
                        all_zxy.append([])
                    slice_ind.append(i_frame)
                    if has_probability:
                        all_prob.append(dict_of_prob_dicts[next_match_key][i0])

                    del dict_of_match_dicts[next_match_key][i0]
                    if verbose >= 3:
                        print(f"Continued tracklet: {next_match_key}, {i0}")

                else:
                    pbar.update(len(slice_ind) - 1)
                    if verbose >= 2:
                        print(f"Tracklet length: {len(slice_ind)}")
                    if verbose >= 3:
                        print(f"Tracklet ended on frame pair: {next_match_key}")
                    break
            else:
                pbar.update(len(slice_ind) - 1)
                if verbose >= 2:
                    print(f"Tracklet reached the end of the recording")

            # Save these lists in the dataframe
            slice_ind = [s + slice_offset for s in slice_ind]
            # TODO: probability here means matches, not neuron detection... change?
            if has_probability:
                all_prob.append(0)  # Make the probability match the zxy
                all_prob = np.expand_dims(np.array(all_prob), axis=-1)
            if verbose >= 2:
                print(f"Ended tracklet with length {len(slice_ind)}")
            df = pd.DataFrame(dict(clust_ind=clust_ind, all_ind_local=[all_ind_local], all_xyz=[all_zxy],
                                   all_prob=[all_prob], slice_ind=[slice_ind]))
            all_dfs.append(df)
            clust_ind += 1

    final_df = pd.concat(all_dfs, ignore_index=True, axis=0)

    return final_df


def fix_global2tracklet_full_dict(df_tracklets, global2tracklet) -> Dict[str, List[str]]:
    return {key: fix_matches_to_use_keys_not_int(df_tracklets, val) for key, val in global2tracklet.items()}


def fix_matches_to_use_keys_not_int(df_tracklet, these_tracklet_ind):
    # Extract the tracklets belonging to this neuron
    isstr = lambda s: isinstance(s, str)
    if all(map(isstr, these_tracklet_ind)):
        # logging.info("Assuming that tracklet matches are proper DataFrame keys")
        # logging.debug(f"First key: {these_tracklet_ind[0]}")
        these_tracklet_names = list(set(these_tracklet_ind))
    else:
        raw_tracklet_names = get_names_from_df(df_tracklet)
        these_tracklet_names = []
        for i_or_name in these_tracklet_ind:
            if i_or_name in these_tracklet_names:
                # TODO: why do I sometimes have problems with non-unique names here?
                # logging.debug(f"Skipping repeated name: {i_or_name} (current names: {these_tracklet_names})")
                continue
            if isstr(i_or_name):
                these_tracklet_names.append(i_or_name)
            else:
                these_tracklet_names.append(raw_tracklet_names[i_or_name])
        # these_tracklet_names = list(set(these_tracklet_names))
    if len(these_tracklet_names) < len(these_tracklet_ind):
        logging.debug(f"Removed repeated indices ({len(these_tracklet_ind)} -> {len(these_tracklet_names)})")
        logging.debug(f"Original list: {these_tracklet_ind}")
    assert all(map(isstr, these_tracklet_names)), f"Not all elements are strings: {these_tracklet_names}"
    return these_tracklet_names


def get_time_overlap_of_candidate_tracklet(candidate_tracklet_name, current_tracklet_names, df_tracklets)\
        -> Optional[Dict[str, Set[int]]]:
    """
    Get the time overlap of a candidate tracklet with the tracklets currently assigned to a neuron

    Parameters
    ----------
    candidate_tracklet_name
    current_tracklet_names
    df_tracklets

    Returns
    -------

    """
    df_candidate_tracklet = df_tracklets[candidate_tracklet_name]
    times_candidate_tracklet = df_candidate_tracklet['z'].dropna().index
    overlap_dict = {}
    for name in current_tracklet_names:
        current_tracklet = df_tracklets[name]
        times_current_tracklet = current_tracklet['z'].dropna().index
        intersection = times_candidate_tracklet.intersection(times_current_tracklet)
        if len(intersection) > 0:
            overlap_dict[name] = intersection
    if len(overlap_dict) == 0:
        return None
    else:
        return overlap_dict


def get_next_name_tracklet_or_neuron(df, name_mode='tracklet'):
    all_names = get_names_from_df(df)
    max_int = name2int_neuron_and_tracklet(all_names[-1])
    # Really want to make sure we are after all other names
    i_tracklet = max_int + 1
    new_name = int2name_using_mode(i_tracklet, name_mode)
    assert new_name not in all_names, "Failed to generate new tracklet name"
    return new_name


def split_tracklet_within_dataframe(all_tracklets, i_split, old_name, verbose=1):
    left_name = old_name
    this_tracklet = all_tracklets[[left_name]]
    idx = this_tracklet.index[this_tracklet[left_name]['z'].notnull()]
    # if i_split not in this_tracklet.dropna(axis=0).index:
    if i_split not in idx:
        logging.warning(f"Tried to split {old_name} at {i_split}, but it doesn't exist at that time")
        return False, all_tracklets, left_name, None
    # Split
    left_half, right_half = split_single_tracklet(i_split, this_tracklet)
    right_name = get_next_name_tracklet_or_neuron(all_tracklets)
    right_half.rename(columns={left_name: right_name}, level=0, inplace=True)
    if verbose >= 1:
        print(f"Creating new tracklet {right_name} from {left_name} by splitting at t={i_split}")
        print(
            f"New non-nan lengths: new: {right_half[right_name]['z'].count()}, old:{left_half[left_name]['z'].count()}")
        # Performance tests
    # all_tracklets = pd.concat([all_tracklets, right_half], axis=1, copy=False, sort=False)
    all_tracklets = all_tracklets.join(right_half, sort=False)
    # all_tracklets = all_tracklets.merge(right_half, sort=False, left_index=True, right_index=True)
    # all_tracklets.assign(**right_half.to_dict())
    # all_tracklets.append(right_half)
    all_tracklets[left_name] = left_half[left_name]
    return True, all_tracklets, left_name, right_name


def split_tracklet_within_sparse_dataframe(all_tracklets, i_split, old_name, verbose=1):
    left_name = old_name
    this_tracklet = all_tracklets[[left_name]]
    idx = this_tracklet.index[this_tracklet[left_name]['z'].notnull()]
    if i_split not in idx:
        logging.warning(f"Tried to split {old_name} at {i_split}, but it doesn't exist at that time")
        return False, all_tracklets, left_name, None
    # Split
    left_half, right_half = split_single_sparse_tracklet(i_split, this_tracklet)
    right_name = get_next_name_tracklet_or_neuron(all_tracklets)
    right_half.rename(columns={left_name: right_name}, level=0, inplace=True)
    if verbose >= 1:
        print(f"Creating new tracklet {right_name} from {left_name} by splitting at t={i_split}")
        print(
            f"New non-nan lengths: new: {right_half[right_name]['z'].count()}, old:{left_half[left_name]['z'].count()}")
    all_tracklets = all_tracklets.join(right_half, sort=False)
    all_tracklets[left_name] = left_half[left_name]
    return True, all_tracklets, left_name, right_name


def split_single_tracklet(i_split, this_tracklet: pd.DataFrame):
    left_half = this_tracklet.copy()
    right_half = this_tracklet.copy()
    left_half.iloc[i_split:] = np.nan
    right_half.iloc[:i_split] = np.nan
    return left_half, right_half


def split_all_tracklets_at_once(df_tracklets: pd.DataFrame, split_list_dict: dict,
                                include_unsplit=True, name_mode='tracklet', name_gen=None, verbose=0):
    """
    Takes a dataframe (should be sparse) and a dictionary with lists of times to split each tracklet

    split_list_dict should have format:
        split_list_dict[tracklet_name] = list_of_times

    Performs one large concatenation at the end

    See: split_using_dict_of_points

    Parameters
    ----------
    df_tracklets
    split_list_dict
    include_unsplit
    name_mode
    name_gen
    verbose

    Returns
    -------

    """
    # Make long list of all split tracklets
    all_new_tracklets = []
    all_unsplit_tracklets = []
    if include_unsplit:
        all_names = get_names_from_df(df_tracklets)
    else:
        all_names = list(split_list_dict.keys())

    name_mapping = {}
    if name_gen is None:
        name_gen = get_next_name_generator(df_tracklets, name_mode)
    all_new_names = []

    for name in tqdm(all_names, leave=False):
        these_splits = split_list_dict.get(name, [])
        this_tracklet = df_tracklets[[name]]
        if len(these_splits) >= 1:
            new_tracklets_list = split_multiple_tracklets(this_tracklet, these_splits)
            all_new_tracklets.extend(new_tracklets_list)
            # Also generate new names, and save them in a mapping
            these_new_tracklet_names = [n for _, n in zip(range(len(new_tracklets_list)), name_gen)]
            name_mapping[name] = these_new_tracklet_names
            all_new_names.extend(these_new_tracklet_names)
        elif include_unsplit:
            all_unsplit_tracklets.append(this_tracklet)

    # Generate new tracklet names, keeping the old ones
    old_names = [get_names_from_df(t)[0] for t in all_unsplit_tracklets]
    # name_gen = get_next_name_generator(df_tracklets, name_mode)
    # new_names = [name for _, name in zip(range(len(all_new_tracklets)), name_gen)]

    # for old_name in old_names:
    #
    # name_mapping = {old_name: name for old_name, name in zip(old_names, name_gen)}

    all_new_tracklets.extend(all_unsplit_tracklets)
    all_new_names.extend(old_names)

    # Also have to remove the old names, i.e. remove the top level of the hierarchy
    all_new_tracklets = [t[get_names_from_df(t)[0]] for t in all_new_tracklets]

    if verbose >= 1:
        logging.info("Concatenating final dataframe, may take a minute")
        logging.info(f"Split {len(all_names)} tracklets to {len(all_new_tracklets)} tracklets")
    df_final = pd.concat(all_new_tracklets, axis=1, keys=all_new_names)
    return df_final, all_new_tracklets, name_mapping


def remove_tracklets_from_dictionary_without_database_match(df_tracklets, global2tracklet):
    # Remove any tracklets from this dict that were dropped in the above steps
    remaining_tracklets = set(get_names_from_df(df_tracklets))
    global2tracklet_new = {}
    for k, v in tqdm(global2tracklet.items()):
        previous_tracklets = set(v)
        current_tracklets = previous_tracklets.intersection(remaining_tracklets)
        removed_tracklets = previous_tracklets - current_tracklets
        if len(removed_tracklets) > 0:
            logging.warning(f"Removed {len(removed_tracklets)} missing tracklets from {k}") #: {removed_tracklets}")
        global2tracklet_new[k] = list(current_tracklets)
    return global2tracklet_new


def check_for_unmatched_tracklets(df_tracklets, global2tracklet, raise_error=True):
    # Remove any tracklets from this dict that were dropped in the above steps
    remaining_tracklets = set(get_names_from_df(df_tracklets))
    unmatched_tracklets = {}
    num_unmatched = 0
    for k, v in global2tracklet.items():
        previous_tracklets = set(v)
        current_tracklets = previous_tracklets.intersection(remaining_tracklets)
        removed_tracklets = previous_tracklets - current_tracklets
        unmatched_tracklets[k] = removed_tracklets
        num_unmatched += len(removed_tracklets)

    if num_unmatched > 0:
        logging.warning(f"Found {num_unmatched} unmatched tracklets")
        if raise_error:
            raise AnalysisOutOfOrderError("3b")
    else:
        print("Tracklet database and matching dictionary are not desynced")

    return unmatched_tracklets