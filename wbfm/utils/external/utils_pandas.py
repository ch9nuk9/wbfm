import logging
from collections import defaultdict
from typing import Tuple, List, Union, Dict

import numpy as np
import pandas as pd


def fix_extra_spaces_in_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_df = {}
    for col in df.columns:
        if isinstance(col, str):
            new_col = col.strip()
        else:
            new_col = col
        new_df[new_col] = df[col]
    return pd.DataFrame(new_df)


def dataframe_to_dataframe_zxy_format(df_tracklets: pd.DataFrame, flip_xy=False) -> pd.DataFrame:
    """Currently, flipxy is true when calling from napari"""
    if not flip_xy:
        coords = ['z', 'x', 'y']
    else:
        coords = ['z', 'y', 'x']
    df_tracklets = df_tracklets.loc(axis=1)[:, coords]
    df_tracklets = df_tracklets.sort_index(axis=1, level=0, sort_remaining=False)
    return df_tracklets


def dataframe_to_numpy_zxy_single_frame(df_tracklets: pd.DataFrame, t: int, flip_xy=False) -> np.ndarray:
    df_zxy = dataframe_to_dataframe_zxy_format(df_tracklets.iloc[[t], :], flip_xy)
    return df_zxy.to_numpy().reshape(-1, 3)


def get_names_of_conflicting_dataframes(tracklet_list: list,
                                        tracklet_network_names: List[str]) -> \
                                        List[List[str]]:
    """
    Given a list of tracklets (sparse pd.Series), determine which if any have overlapping indices

    tracklet_network_names is used to extract the final names

    Parameters
    ----------
    tracklet_list
    tracklet_network_names

    Returns
    -------

    """
    all_times = [t.dropna().index for t in tracklet_list]
    # overlapping_tracklet_ind = []
    overlapping_tracklet_names = []
    for i_base_neuron, times_base_neuron in enumerate(all_times):
        this_overlapping_ind = [i_base_neuron]
        # This base neuron could have many overlaps; keep track of all of them
        for i2, times_target_neuron in enumerate(all_times[i_base_neuron + 1:]):
            if len(times_base_neuron.intersection(times_target_neuron)) > 0:
                i_target_neuron = i2 + i_base_neuron + 1
                this_overlapping_ind.append(i_target_neuron)
        if len(this_overlapping_ind) > 1:
            # overlapping_tracklet_ind.append(this_overlapping_ind)
            these_names = [tracklet_network_names[n] for n in this_overlapping_ind]
            overlapping_tracklet_names.append(these_names)
    return overlapping_tracklet_names


def get_times_of_conflicting_dataframes(tracklet_list: List[Union[pd.Series, pd.DataFrame]],
                                        tracklet_network_names: List[str],
                                        verbose=0) -> Dict[str, List[int]]:
    """
    Takes a list of tracklets and their names, and finds the conflict points between the tracklets in the list

    tracklet_list and tracklet_network_names should be synchronized lists

    Returns a dictionary of the times of the conflict points, indexed by the conflicting tracklet names

    In principle this function is symmetric with respect to tracklet_list and tracklet_network_names, but
    it is not exactly symmetric with respect to the order of the tracklets in the list. This is because the
    conflict points are determined by the order of the tracklets in the list.

    Parameters
    ----------
    tracklet_list
    tracklet_network_names
    verbose

    Returns
    -------

    """
    all_indices = [t.dropna().index for t in tracklet_list]
    overlapping_tracklet_conflict_points = defaultdict(list)
    known_empty_tracklets = set()
    for i1, (idx1, base_tracklet_name) in enumerate(zip(all_indices, tracklet_network_names)):
        if base_tracklet_name in known_empty_tracklets:
            continue
        if len(idx1) == 0:
            logging.warning(f"Skipping empty tracklet {base_tracklet_name}")
            known_empty_tracklets.add(base_tracklet_name)
            continue
        idx1_edges = [int(idx1[0]), int(idx1[-1])+1]
        for idx2, target_tracklet_name in zip(all_indices[i1 + 1:], tracklet_network_names[i1 + 1:]):
            if base_tracklet_name == target_tracklet_name:
                logging.warning("Attempted to compare tracklet to itself; continuing")
                continue
            if target_tracklet_name in known_empty_tracklets:
                continue
            if len(idx2) <= 1:
                # logging.warning("Tracklet was empty; continuing")
                known_empty_tracklets.add(target_tracklet_name)
                continue
            intersecting_ind = list(idx1.intersection(idx2))
            idx2_edges = [int(idx2[0]), int(idx2[-1])+1]
            if len(intersecting_ind) > 0:
                intersecting_ind.sort()
                conflict_points = [intersecting_ind[0], intersecting_ind[-1]+1]
                # Want to split both tracklets at both points, if they aren't at the extreme
                for c in conflict_points:
                    if c not in idx1_edges:
                        overlapping_tracklet_conflict_points[base_tracklet_name].append(c)
                        if verbose >= 2:
                            print(f"Added point {c} to {base_tracklet_name}")
                    if c not in idx2_edges:
                        overlapping_tracklet_conflict_points[target_tracklet_name].append(c)
                        if verbose >= 2:
                            print(f"Added point {c} to {target_tracklet_name}")
                    if verbose >= 2:
                        print(f"Added point {c} with edge indices {idx1_edges} and {idx2_edges}")

    return overlapping_tracklet_conflict_points


def check_if_fully_sparse(df: pd.DataFrame) -> False:
    """
    Checks if each individual column of df is sparse

    Note: does not scale well!
    """
    # No good way: https://github.com/pandas-dev/pandas/issues/26706
    return df.dtypes.apply(pd.api.types.is_sparse).all()


def to_sparse_multiindex(df: pd.DataFrame, new_columns=None):
    """
    Converts a dataframe to a fully sparse version that may have renamed columns

    Must be done in a loop, per column (note: column index will generally be a tuple)

    Parameters
    ----------
    df
    new_columns

    Returns
    -------

    """
    if new_columns is None:
        new_columns = df
    new_columns = new_columns.astype(pd.SparseDtype("float", np.nan))  # This works, but then direct assignment doesn't
    for c in new_columns.columns:
        df[c] = new_columns[c]

    return df


def ensure_dense_dataframe(df: pd.DataFrame, new_columns=None):
    """
    Converts a dataframe to a fully dense version

    Must be done in a loop, per column (note: column index will generally be a tuple)

    Parameters
    ----------
    df
    new_columns

    Returns
    -------

    """
    if new_columns is None:
        new_columns = df
    for c in new_columns.columns:
        try:
            df[c] = new_columns[c].sparse.to_dense()
        except AttributeError:
            df[c] = new_columns[c]

    return df


def cast_int_or_nan(i: Union[list, int]):
    """Cast as integer, but do not crash if np.nan"""
    if isinstance(i, (list, pd.Series)):
        return [cast_int_or_nan(_i) for _i in i]
    if np.isnan(i):
        return i
    else:
        return int(i)


def legacy_get_contiguous_blocks_from_column(column_or_series: pd.Series, already_boolean=False,
                                      include_end_if_censored=True, skip_boolean_check=False,
                                      DEBUG=False) -> Tuple[list, list]:
    """
    Slower pandas-only version of get_contiguous_blocks_from_column

    Parameters
    ----------
    column_or_series
    already_boolean
    include_end_if_censored: include the last index if a block is still present.
        Otherwise, len(starts) may be less than len(ends)
    skip_boolean_check


    Returns
    -------

    """
    if already_boolean:
        if not skip_boolean_check:
            assert len(np.unique(column_or_series)) <= 2, "Vector must be actually boolean"
        bool_column_or_series = column_or_series
    else:
        bool_column_or_series = column_or_series.isnull()

    if hasattr(column_or_series, 'sparse'):
        change_ind = np.where(bool_column_or_series.sparse.to_dense().diff().values)[0]
    else:
        change_ind = np.where(bool_column_or_series.diff().values)[0]
    if DEBUG:
        print(change_ind)
        print(bool_column_or_series.iloc[change_ind])

    block_starts = []
    block_ends = []
    for i in change_ind:
        if np.isnan(column_or_series.iat[i]) or (already_boolean and not bool_column_or_series.iat[i]):
            if i > 0:
                # Diff always has a value here, but it can only be a start, not an end
                block_ends.append(i)
        else:
            if not already_boolean or bool_column_or_series.iat[i]:
                block_starts.append(i)
    if include_end_if_censored and len(block_ends) < len(block_starts):
        block_ends.append(len(bool_column_or_series))
    return block_starts, block_ends


def get_contiguous_blocks_from_column(column_or_series: pd.Series, already_boolean=False,
                                      include_end_if_censored=True, skip_boolean_check=False,
                                      DEBUG=False) -> Tuple[list, list]:
    """
    Given a pd.Series that may have gaps, get the indices of the contiguous blocks of non-nan points

    Parameters
    ----------
    column_or_series
    already_boolean
    include_end_if_censored: include the last index if a block is still present.
        Otherwise, len(starts) may be less than len(ends)
    skip_boolean_check


    Returns
    -------

    """
    if already_boolean:
        if not skip_boolean_check:
            assert len(np.unique(column_or_series)) <= 2, "Vector must be actually boolean"
        bool_column_or_series = column_or_series
    else:
        bool_column_or_series = column_or_series.isnull()

    if hasattr(column_or_series, 'sparse'):
        bool_values = bool_column_or_series.sparse.to_dense().to_numpy()
    else:
        bool_values = bool_column_or_series.to_numpy()

    # Align with pandas version
    change_ind = np.where(np.diff(bool_values))[0]
    change_ind = np.hstack([0, change_ind + 1])
    if DEBUG:
        print(change_ind)
        print(bool_values[change_ind])

    block_starts = []
    block_ends = []
    for i in change_ind:
        if np.isnan(bool_values[i]) or (already_boolean and not bool_values[i]):
            if i > 0:
                # Pandas diff always has a value here, but it can only be a start, not an end
                block_ends.append(i)
        else:
            if not already_boolean or bool_values[i]:
                block_starts.append(i)
    if include_end_if_censored and len(block_ends) < len(block_starts):
        block_ends.append(len(bool_column_or_series))
    return block_starts, block_ends


def extend_short_states(starts, ends, max_len, state_length_minimum=10, DEBUG=False):
    """
    Given a binary Series and the contiguous blocks of True values, extend short blocks by a certain amount

    Parameters
    ----------
    starts
    ends
    state_length_minimum
    DEBUG

    Returns
    -------

    """
    new_ends = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        current_duration = end - start
        if current_duration < state_length_minimum:
            if DEBUG:
                print("Extending state")
            # Check to make sure the states we're extending into are allowed, i.e. not the edge or True
            new_end = min(max_len, end + state_length_minimum)
            if i < len(ends) - 1:
                if new_end > starts[i + 1]:
                    new_end = starts[i + 1] - 1
                    if DEBUG:
                        print("Clipping to next start")
            new_ends.append(new_end)
        else:
            new_ends.append(end)
    return starts, new_ends

def get_relative_onset_times_from_two_binary_vectors(column: pd.Series, sub_column: pd.Series):
    """
    Assumes the two columns are binary, and that sub_column has events which are contained within the events of column

    Parameters
    ----------
    column
    sub_column

    Returns
    -------

    """

    starts, ends = get_contiguous_blocks_from_column(column, already_boolean=True)
    sub_starts, sub_ends = get_contiguous_blocks_from_column(sub_column, already_boolean=True)

    relative_onset_times = []
    for start, end in zip(starts, ends):
        for sub_start, sub_end in zip(sub_starts, sub_ends):
            if sub_start >= start and sub_end <= end:
                relative_onset_times.append(sub_start - start)
                # Remove this sub event so it can't be used again
                sub_starts.remove(sub_start)
                sub_ends.remove(sub_end)
                break

    return relative_onset_times


def calc_eventwise_cooccurrence_matrix(df: pd.DataFrame, column1: str, column2: str, column_state: str,
                                       normalize=True,
                                       DEBUG=False):
    """
    Given a dataframe with two binary columns, calculate a co-occurrence matrix per event.
    An event is defined as a contiguous block of True values in either column1 or column2

    Parameters
    ----------
    df
    column1
    column2
    normalize
    DEBUG

    Returns
    -------

    """
    if DEBUG:
        print(df[column1].value_counts())
        print(df[column2].value_counts())

    # Get the contiguous blocks of True values in each column
    starts1, ends1 = get_contiguous_blocks_from_column(df[column1], already_boolean=True, DEBUG=DEBUG)
    starts2, ends2 = get_contiguous_blocks_from_column(df[column2], already_boolean=True, DEBUG=DEBUG)
    # Get the contiguous blocks of the state to check (may be same as column1 or column2)
    starts_state, ends_state = get_contiguous_blocks_from_column(df[column_state], already_boolean=True, DEBUG=DEBUG)

    # Get the co-occurrence matrix for each event
    cooccurrence_matrices = []
    for start_state, end_state in zip(starts_state, ends_state):
        # Do not need indices, just classify each state as one of the 4 cases
        # 0: no overlap, 1: column1 overlap, 2: column2 overlap, 3: both overlap
        # Check each event type independently
        col1_overlap = False
        col2_overlap = False
        for start1, end1 in zip(starts1, ends1):
            if (start1 <= start_state <= end1) or (start_state <= start1 <= end_state):
                # column1 overlaps
                col1_overlap = True
                break
        for start2, end2 in zip(starts2, ends2):
            # Check if the two events overlap
            if (start2 <= start_state <= end2) or (start_state <= start2 <= end_state):
                # Only column2 overlaps
                col2_overlap = True
                break
        this_df = pd.DataFrame([[col1_overlap, col2_overlap]], columns=[column1, column2])
        cooccurrence_matrices.append(this_df)

    # Combine the co-occurrence matrices
    cooccurrence_matrix = pd.concat(cooccurrence_matrices, axis=0, ignore_index=True)
    # cooccurrence_matrix = pd.DataFrame(cooccurrence_matrices)
    return cooccurrence_matrix


def calc_surpyval_durations_and_censoring(all_starts, all_ends):
    """
    Uses the conventions of the surpyval package and checks for censoring
    More detail: https://surpyval.readthedocs.io/en/latest/Types%20of%20Data.html#censored-data

    Quote:
    The possible values of c are -1, 0, 1, and 2.
    The convention tries to illustrate the concept of left, right, and interval censoring on the timeline.
    That is, -1 is the flag for left censoring because it is to the left of an observed failure.
    With an observed failure at 0. 1 is used to flag a value as right censored.
    Finally, 2 is used to flag a value as being intervally censored because it has 2 data points, a left and right point.

    Example:
    import surpyval

    x = [3, 3, 3, 4, 4, [4, 6], [6, 8], 8]
    c = [-1, -1, -1, 0, 0, 2, 2, 1]

    model = surpyval.Weibull.fit(x=x, c=c)

    Parameters
    ----------
    all_starts
    all_ends

    Returns
    -------

    """

    censored_vec = []
    duration_vec = []
    for i, (s, e) in enumerate(zip(all_starts, all_ends)):
        duration_vec.append(e - s)
        # For this type of data, there won't be any left censoring, only right censoring at both edges
        if s == 0 or i == len(all_starts) - 1:
            censored_vec.append(1)
        else:
            censored_vec.append(0)
    return duration_vec, censored_vec


def remove_short_state_changes(bool_column: pd.Series, min_length, only_replace_these_states=None,
                               replace_with_next_state=True, depth=0, remove_small_false_first=True,
                               DEBUG=False) -> pd.Series:
    """
    Removes very small states from an integer series, assuming they are noise. Replaces the tiny states with the
    surrounding state index. If the before and after are not the same, chooses based on 'replace_with_preceding_state'

    Note: bool_column should actually be boolean

    Parameters
    ----------
    only_replace_these_states: Optional; only replace states of a certain index
    bool_column
    min_length
    replace_with_next_state

    Returns
    -------

    """
    # Apply this function to both the column and its inverse
    if depth == 0:
        kwargs = dict(min_length=min_length, only_replace_these_states=only_replace_these_states,
                      replace_with_next_state=replace_with_next_state, depth=1, DEBUG=DEBUG)
        # Do two steps: remove small True and False
        # Decide which to keep based on the remove_small_false_first parameter
        if remove_small_false_first:
            small_false_removed = ~remove_short_state_changes(~bool_column, **kwargs)
            final_vec = remove_short_state_changes(small_false_removed, **kwargs)
        else:
            small_true_removed = remove_short_state_changes(bool_column, **kwargs)
            final_vec = remove_short_state_changes(small_true_removed, **kwargs)

        return final_vec

    starts, ends = get_contiguous_blocks_from_column(bool_column, already_boolean=True)
    new_column = bool_column.copy()

    # Compare the end to the next start, i.e. the gap between states
    if DEBUG:
        print(starts, ends)
    for next_start, previous_end in zip(starts[1:], ends[:-1]):
        if DEBUG:
            print("Checking: ", next_start, previous_end)
        if next_start - previous_end < min_length:
            if only_replace_these_states is None or bool_column.iat[next_start] in only_replace_these_states:
                # Beginning and end are special
                if previous_end >= len(new_column) or (replace_with_next_state and next_start > 0):
                    replacement_state = bool_column.iat[next_start]
                else:
                    replacement_state = bool_column.iat[previous_end - 1]
                if DEBUG:
                    print("Replacing: ", next_start, previous_end, replacement_state)
                new_column.iloc[previous_end:next_start] = replacement_state
    return new_column


def get_durations_from_column(column_or_series: pd.Series, already_boolean=False, remove_edges=True) -> list:
    """
    Given a pd.Series that may have gaps, get the durations of contiguous blocks of non-nan points

    If a specific state is desired, pass 'column_or_series==val' instead of 'column_or_series'

    See also get_durations_from_column

    Parameters
    ----------
    remove_edges: remove events if they touch the edges (these lengths have a censoring effect)
    already_boolean
    column_or_series

    Returns
    -------

    """

    block_starts, block_ends = get_contiguous_blocks_from_column(column_or_series, already_boolean=already_boolean)
    durations = []
    for e, s in zip(block_ends, block_starts):
        if remove_edges and (s == 0 or e == len(column_or_series)):
            continue
        durations.append(e - s)

    return durations


def df_to_matches(df_gt: pd.DataFrame, t0: int, t1: int = None, col='raw_neuron_ind_in_list') -> list:
    """
    Converts a dataframe that has a column corresponding to an ID index into a list of matches between indices

    Parameters
    ----------
    df_gt
    t0
    t1
    col

    Returns
    -------

    """
    if t1 is None:
        t1 = t0 + 1

    ind0 = df_gt.loc[t0, (slice(None), col)]
    ind1 = df_gt.loc[t1, (slice(None), col)]

    def _neither_nan(i0, i1):
        return (not np.isnan(i0)) and (not np.isnan(i1))

    return list([int(i0), int(i1)] for i0, i1 in zip(ind0, ind1) if _neither_nan(i0, i1))


def accuracy_of_matches(gt_matches, new_matches, null_value=-1, allow_unknown=True):
    """
    Expects a list of 2-element lists for the matches

    if allow_unknown is False, then:
    Assumes that the gt is complete, i.e. there are no extra matches in new_matches that might be correct but unknown
    """
    tp = 0
    fp = 0
    unknown = 0
    try:
        gt_dict = dict(np.array(gt_matches))
    except ValueError:
        gt_dict = dict(np.array(gt_matches[0]))
    for m in new_matches:
        gt_val = gt_dict.get(m[0], None)
        if allow_unknown and gt_val is None:
            unknown += 1
        elif m[1] == gt_val:
            tp += 1
        else:
            fp += 1
    # Add the remainder (unmatched in the model) to fn
    fn = len(gt_dict) - tp
    return tp, fp, fn, unknown


def fill_missing_indices_with_nan(df: Union[pd.DataFrame, pd.Series], expected_max_t=None) -> \
        Tuple[Union[pd.DataFrame, pd.Series], int]:
    """
    Given a dataframe that may skip time points (e.g. the Index is 1, 2, 5), fill the missing Index values with nan

    Parameters
    ----------
    expected_max_t
    df

    Returns
    -------

    """
    t = df.index
    dfs_to_add = []
    # Check if df is a series or dataframe
    is_series = isinstance(df, pd.Series)

    if len(t) != int(t[-1]) + 1:
        add_indices = pd.Index(range(int(t[-1]))).difference(t)
        if is_series:
            df_interleave = pd.Series(index=add_indices)
        else:
            df_interleave = pd.DataFrame(index=add_indices, columns=df.columns)
        dfs_to_add.append(df_interleave)
        num_added = df_interleave.shape[0]
    else:
        num_added = 0

    current_max_t = df.shape[0] + num_added
    if expected_max_t is not None and current_max_t != expected_max_t:
        end_indices = pd.Index(range(current_max_t, expected_max_t))
        if is_series:
            df_nan_at_end = pd.Series(index=end_indices)
        else:
            df_nan_at_end = pd.DataFrame(index=end_indices, columns=df.columns)
        dfs_to_add.append(df_nan_at_end)
        num_added += df_nan_at_end.shape[0]

    if dfs_to_add is not None:
        dfs_to_add.append(df)
        df = pd.concat(dfs_to_add).sort_index()
    return df, num_added


def get_column_name_from_time_and_column_value(df: pd.DataFrame, i_time: int, col_value: int, col_name: str):
    """
    Note that the col_value could be other data types as well

    Parameters
    ----------
    df
    i_time
    col_value
    col_name

    Returns
    -------

    """
    mask = df.loc[i_time, (slice(None), col_name)] == col_value
    try:
        ind = np.where(mask)[0][0]
        return ind, mask.index.get_level_values(0)[mask][0]
    except IndexError:
        return None, None


def correlate_return_cross_terms(df0: pd.DataFrame, df1: pd.DataFrame = None) -> pd.DataFrame:
    """
    Like df.corr(), but acts on two dataframes, returning only the cross terms

    Parameters
    ----------
    df0
    df1

    Returns
    -------

    """
    if df1 is None:
        # This is already build-in to pandas
        return df0.corr()
    df_corr = pd.concat([df0, df1], axis=1).corr()
    return get_corner_from_corr_df(df0, df_corr)


def get_corner_from_corr_df(df0: pd.DataFrame, df_corr: pd.DataFrame):
    """
    If correlations are calculated between a concatenated version of df_trace and something else, this gets only the
    cross terms from df_corr

    Parameters
    ----------
    df0
    df_corr

    Returns
    -------

    """
    ind_nonneuron = np.arange(df0.shape[1], df_corr.shape[1])
    ind_neurons = np.arange(0, df0.shape[1])
    return df_corr.iloc[ind_neurons, ind_nonneuron]


def melt_nested_dict(nested_dict: dict, all_same_lengths=True,
                     new_column_name='dataset_name'):
    """
    Assuming a nested dict of the form:
    {'key_outer': {'key_inner': vector}}

    Returns a dataframe of the form:
    key_name | key_outer
    key_inner, val1
    key_inner, val2
    ...

    i.e. the inner key is repeated in a new column, and the vector is expanded into a column with the same name as the
    outer key

    Parameters
    ----------
    nested_dict

    Returns
    -------

    """

    all_melted_dfs = []
    for i, (k, v) in enumerate(nested_dict.items()):
        if all_same_lengths:
            df_square = pd.DataFrame(v)
        else:
            df_square = pd.DataFrame.from_dict(v, orient='index').T
        two_col_dataframe = df_square.melt()
        two_col_dataframe.astype({'variable': str})
        two_col_dataframe.columns = [new_column_name, k]
        if i > 0:
            two_col_dataframe.drop(columns=[new_column_name], inplace=True)
        all_melted_dfs.append(two_col_dataframe)

    df_melted = all_melted_dfs[0].copy()
    for df in all_melted_dfs[1:]:
        # merging on the dataset name somehow doesn't properly match names, so just drop that
        df_melted = df_melted.join(df)
    return df_melted


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens the columns of a dataframe with a multiindex

    Parameters
    ----------
    df

    Returns
    -------

    """
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


def flatten_nested_dict(nested_dict: dict) -> dict:
    """
    Flattens the keys of a nested dictionary

    See flatten_multiindex_columns for a similar function that flattens the columns of a dataframe

    Parameters
    ----------
    nested_dict

    Returns
    -------

    """
    flat_dict = {}
    for outer_key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            keys = [outer_key, inner_key]
            flat_dict['_'.join(keys).strip()] = value
    return flat_dict


def split_flattened_index(flattened_index: list) -> Dict[str, Tuple[str, str]]:
    """
    Attempts to undo the flattening of a nested dictionary or dataframe

    Assumes that the first name is unknown, but that the second name is of the format: 'neuron_XYZ'

    See flatten_multiindex_columns and flatten_nested_dict for opposite functions

    Parameters
    ----------
    nested_dict

    Returns
    -------

    """
    unflattened_dict = {}
    for key in flattened_index:
        if '_neuron_' in key:
            split_str = 'neuron'
        elif '_segment_' in key:
            split_str = 'segment'
        else:
            split_str = None

        if split_str is not None:
            # Split a string like 'ZIM2165_Gcamp7b_worm3-2022-12-05_neuron_054' into
            # ['ZIM2165_Gcamp7b_worm3-2022-12-05', 'neuron_054']
            split_key = key.split(f'_{split_str}_')
            if len(split_key) == 2:
                neuron_id = split_key[1]
                neuron_name = f"{split_str}_{neuron_id}"
                dataset_name = split_key[0]
                unflattened_dict[key] = (dataset_name, neuron_name)
            else:
                raise ValueError(f"Could not split key {key}")
        else:
            # Split a string like 'ZIM2165_Gcamp7b_worm3-2022-12-05_signed_stage_speed' into
            # ['ZIM2165_Gcamp7b_worm3-2022-12-05', 'signed_stage_speed']
            # Or a string like 'ZIM2165_Gcamp7b_worm3-2022-12-05_summed_curvature' into
            # ['ZIM2165_Gcamp7b_worm3-2022-12-05', 'summed_curvature']
            # The name of the actual trace (second name) may have multiple underscores, so this is not consistent
            # But all project names end with a number. Some second names have a number, but not at the beginning
            try:
                split_key = key.split('_')
                has_any_digits = lambda x: any([c.isdigit() for c in x])
                last_num_index = max([i for i, c in enumerate(split_key) if has_any_digits(c)])
                # If the last number is at the end, then split on the second-to-last instead
                if last_num_index == len(split_key) - 1:
                    last_num_index -= 1
                dataset_name = '_'.join(split_key[:last_num_index + 1])
                trace_name = '_'.join(split_key[last_num_index + 1:])
                unflattened_dict[key] = (dataset_name, trace_name)

                # last_num_index = max([i for i, c in enumerate(key) if c.isdigit()])
                # dataset_name = key[:last_num_index + 1]
                # trace_name = key[last_num_index + 2:]  # Assume there is a single character as a splitter
                # unflattened_dict[key] = (dataset_name, trace_name)
            except ValueError as e:
                print(e)
                raise ValueError(f"Could not split key {key}")

    return unflattened_dict


def combine_rows_with_same_suffix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assuming a dataframe with column names that can be split with split_flattened_index, combine rows with the same
    suffix (i.e. second part of the name)

    Parameters
    ----------
    df

    Returns
    -------

    """
    unique_name_dict = defaultdict(list)

    for name in df.columns:
        segment_name = split_flattened_index([name])[name][1]
        unique_name_dict[segment_name].append(df[name])

    df_unique_names = {}
    for name, list_of_triggered_segments in unique_name_dict.items():
        df_unique_names[name] = np.nanmean(np.vstack(list_of_triggered_segments), axis=0)
    df_unique_names = pd.DataFrame(df_unique_names)
    df_unique_names = df_unique_names.T.sort_index().T
    return df_unique_names


def count_unique_datasets_from_flattened_index(flattened_index: list) -> int:
    """
    Counts the number of unique datasets in a flattened index

    Parameters
    ----------
    flattened_index

    Returns
    -------

    """
    unflattened_dict = split_flattened_index(flattened_index)
    unique_datasets = set([unflattened_dict[key][0] for key in unflattened_dict.keys()])
    return len(unique_datasets)


def save_valid_ind_1d_or_2d(df, valid_ind):
    if len(df.shape) == 2:
        df = df.iloc[valid_ind, :]
    elif len(df.shape) == 1:
        df = df.iloc[valid_ind]
    else:
        raise NotImplementedError("Must be 1d or 2d")
    return df


def make_binary_vector_from_starts_and_ends(starts, ends, original_vals, pad_nan_points=0):
    """
    Makes a binary vector from a list of starts and ends

    See get_contiguous_blocks_from_column

    Parameters
    ----------
    starts
    ends
    original_vals
    pad_nan_points

    Returns
    -------

    """

    # Split pad_nan_points if it has different values for starts and ends
    if isinstance(pad_nan_points, (list, tuple)):
        pad_nan_points_start, pad_nan_points_end = pad_nan_points
    else:
        pad_nan_points_start = pad_nan_points_end = pad_nan_points

    idx_boolean = np.zeros_like(original_vals)
    for s, e in zip(starts, ends):
        s = np.clip(s - pad_nan_points_start, a_min=0, a_max=len(original_vals))
        e = np.clip(e + pad_nan_points_end, a_min=0, a_max=len(original_vals))
        idx_boolean[s:e] = 1

    return idx_boolean


def extend_binary_vector(binary_state: pd.Series, alt_binary_state: pd.Series) -> pd.Series:
    starts, ends = get_contiguous_blocks_from_column(binary_state, already_boolean=True)
    _, alt_ends = get_contiguous_blocks_from_column(alt_binary_state, already_boolean=True)
    for i in range(len(ends)):
        # If the next time point is one of the allowed succeeding states, extend the end by replacing it
        # with the next end of the allowed succeeding state
        if binary_state.iat[ends[i]] and alt_binary_state.iat[ends[i] + 1]:
            # The index of the alt state is not generally the same as the index of the state
            # So we have to find the next end of the alt state
            next_end = alt_ends[alt_ends > ends[i]].min()
            # But make sure that it doesn't overlap with the next start of the state
            next_start = starts[starts > ends[i]].min()
            if next_end < next_start:
                ends[i] = next_end
            else:
                ends[i] = next_start - 1
    # Recreate the binary state from the modified ends
    binary_state = pd.Series(make_binary_vector_from_starts_and_ends(starts, ends, len(binary_state)))
    return binary_state


def pad_events_in_binary_vector(vec, pad_length=(1, 1)):
    starts, ends = get_contiguous_blocks_from_column(vec, already_boolean=True)
    vec_padded = make_binary_vector_from_starts_and_ends(starts, ends, vec, pad_nan_points=pad_length)
    return vec_padded


def build_tracks_from_dataframe(df_single_track, likelihood_thresh=None, z_to_xy_ratio=1.0):
    # Just visualize one neuron for now
    # 5 columns:
    # track_id, t, z, y, x
    try:
        coords = ['z', 'x', 'y']
        zxy_array = df_single_track[coords].to_numpy(copy=True)
    except KeyError:
        coords = ['z_dlc', 'x_dlc', 'y_dlc']
        zxy_array = df_single_track[coords].to_numpy(copy=True)

    zxy_array = np.copy(zxy_array)

    all_tracks_list = []
    t_array = np.expand_dims(np.arange(zxy_array.shape[0]), axis=1)

    if likelihood_thresh is not None and 'likelihood' in df_single_track:
        to_remove = df_single_track['likelihood'] < likelihood_thresh
    else:
        to_remove = np.zeros_like(zxy_array[:, 0], dtype=bool)
    zxy_array[to_remove, :] = 0

    # Also remove values that are entirely nan
    rows_not_nan = ~(np.isnan(zxy_array)[:, 0])
    zxy_array = zxy_array[rows_not_nan, :]
    zxy_array[:, 0] *= z_to_xy_ratio
    t_array = t_array[rows_not_nan, :]

    all_tracks_list.append(np.hstack([t_array, zxy_array]))
    all_tracks_array = np.vstack(all_tracks_list)

    track_of_point = np.hstack([np.ones((all_tracks_array.shape[0], 1)), all_tracks_array])

    return all_tracks_array, track_of_point, to_remove


def get_dataframe_of_transitions(state_vector: pd.Series, convert_to_probabilities=False,
                                 ignore_diagonal=False,
                                 transition_observation_threshold=1, state_observation_threshold=None, DEBUG=False):
    """
    Gets the transition dictionary of a state vector, i.e. the number of times each state transition occurs

    Parameters
    ----------
    state_vector
    convert_to_probabilities - if True, converts the counts to probabilities
    ignore_diagonal - if True, sets the diagonal to 0
    state_occupancy_threshold - if not None, removes states that have less than this number of observations
        (note: this is in observation units, not percentage units)

    Returns
    -------

    """
    # Crosstab remembers the original time index if we give it a pd.Series
    if isinstance(state_vector, pd.Series):
        state_vector = state_vector.values
    df_transitions = pd.crosstab(pd.Series(state_vector[:-1], name='from_category'),
                                 pd.Series(state_vector[1:], name='to_category'))

    if state_observation_threshold is not None:
        # Remove individual entries that have less than the threshold number of observations
        bad_entries = df_transitions < state_observation_threshold
        df_transitions[bad_entries] = 0
        if DEBUG:
            print(bad_entries)

    if transition_observation_threshold is not None:
        # Remove rows and columns that have less than the threshold number of observations
        # Should keep the matrix square, so just use the row sums
        good_rows = df_transitions.sum(axis=1) > transition_observation_threshold
        df_transitions = df_transitions.loc[good_rows, good_rows]
        if DEBUG:
            print(good_rows)

    if ignore_diagonal:
        np.fill_diagonal(df_transitions.values, 0)

    if convert_to_probabilities:
        # Note: must use .div because the columns and rows have the same names, thus pandas is confused
        df_transitions = df_transitions.div(df_transitions.sum(axis=1), axis=0)

    return df_transitions


def apply_to_dict_of_dfs_and_concat(dict_of_dfs, func):
    """
    Applies a summary function to a dictionary of dataframes, setting the dict key to a new column

    Intended to create a very tall dataframe to be plotted using plotly

    Parameters
    ----------
    dict_of_dfs

    Returns
    -------

    """

    new_dfs = []
    for name, df in dict_of_dfs.items():
        df = func(df)
        df['name'] = name
        new_dfs.append(df)

    df_concat = pd.concat(new_dfs)
    return df_concat


def combine_columns_with_suffix(df, suffixes=None, how='mean', raw_names_to_keep=None, DEBUG=False):
    """
    Combines columns with the same prefix and different suffixes

    Note: for now there is one hardcoded exception: AQR, which has no AQL partner

    Parameters
    ----------
    df
    suffixes

    Returns
    -------

    """
    if suffixes is None:
        suffixes = ['L', 'R']
    if raw_names_to_keep is None:
        # AQR has no AQL partner
        raw_names_to_keep = {'AQR'}

    dict_df_combined = dict()
    # Loop through columns and check if they have a suffix; if so, search for the other suffix and combine
    base_names_found = set()
    for col in df.columns:
        if col in raw_names_to_keep:
            dict_df_combined[col] = df[col]
            continue
        num_suffixes_found = 0
        col_base = None
        for suffix in suffixes:
            if col.endswith(suffix):
                col_base = col[:-len(suffix)]
                if col_base not in base_names_found:
                    if DEBUG:
                        print(f"Found base name: {col_base} with suffix {suffix}")
                    num_suffixes_found += 1
                    # There could be a column with any other suffixes, so loop again
                    for other_suffix in suffixes:
                        if other_suffix == suffix:
                            continue
                        col_other = col_base + other_suffix
                        if col_other in df.columns:
                            if DEBUG:
                                print(f"Found partner for {col_base}: {col_other}")
                            dict_df_combined[col_base] = df[col] + df[col_other]
                            num_suffixes_found += 1
                else:
                    base_names_found.add(col_base)
        if num_suffixes_found == 0:
            # Then no suffixes were found, so keep the column as is
            if DEBUG:
                print(f"No suffixes found for {col}, keeping as is")
            dict_df_combined[col] = df[col]
        elif num_suffixes_found == 1:
            # Then nothing to combine, just keep the original time series with the new base name
            assert col_base is not None
            dict_df_combined[col_base] = df[col]
        elif num_suffixes_found > 1:
            # Then we have combined the columns, and need to finish the operation
            if how == 'mean':
                dict_df_combined[col_base] /= num_suffixes_found
                if DEBUG:
                    print(f"Dividing {col_base} by {num_suffixes_found}")
            elif how == 'sum':
                pass
            else:
                raise NotImplementedError(f"how={how} not implemented (valid='mean' and 'sum')")
        else:
            # Should not happen
            raise NotImplementedError

    # Reinitialize the dataframe to fix fragmentation... but doesn't actually work :/
    df_combined = pd.DataFrame(dict_df_combined)

    return df_combined


def fill_gaps_categorical(x, window):
    """Fill gaps in a categorical series using a rolling window median on the factorized values"""
    # From: https://stackoverflow.com/questions/70551614/python-pandas-most-common-value-over-rolling-window
    # Factorize
    y, label = pd.factorize(x)
    y = pd.Series(y)
    # Replace the nan factorization with nan
    y.replace(-1, np.nan, inplace=True)
    label = pd.Series(label)
    # Correct values
    y = y.rolling(window=window, min_periods=1, center=True).median().round()
    # Unfactorize
    y = y.map(label)
    return y


def combine_indices_categorical(x):
    """Combine indices of a categorical series using a rolling window median on the factorized values"""
    # Factorize
    y, label = pd.factorize(x)
    y = pd.Series(y, index=x.index)
    label = pd.Series(label)
    # Correct values (combine indices)
    y = y.groupby(y.index).median().round()
    # Unfactorize
    y = y.map(label)
    return y


def resample_categorical(x, target_len=100):
    """
    Resample a categorical series to a target length using either:
    1. combine_indices_categorical if downsampling
    2. fill_gaps_categorical if upsampling
    """
    # Create the output series
    x_new = pd.Series(index=np.arange(target_len))
    # Set the index of the input to be fractions of the target, but round
    x_old = pd.Series(x)
    new_index = np.round(np.linspace(0, target_len - 1, len(x))).astype(int)
    x_old.index = new_index

    # If we are downsampling, then there will be duplicate indices
    if len(x) > target_len:
        x_new = combine_indices_categorical(x_old)
    else:
        # Update the new x with the rounded values
        x_new.update(x_old)

        # Interpolate using regular rolling mean
        x_new = fill_gaps_categorical(x_new, window=target_len)

    return list(x_new)


def get_contiguous_blocks_from_two_columns(df, col_group, col_value):
    """
    Somewhat similar to get_contiguous_blocks_from_column, but directly groups and returns the values of another column

    Parameters
    ----------
    df
    col_group
    col_value

    Returns
    -------

    """
    df = df.copy()

    # Create a grouping variable for each contiguous block of 'A'
    grouping_variable = (df[col_group] != df[col_group].shift()).cumsum()

    # Group by the new grouping variable and apply the function
    result = df.groupby([col_group, grouping_variable])[col_value].apply(list)
    result.index.set_names((col_group, col_value), inplace=True)

    return result


def calc_closest_index(index, idx_target, return_idx=True, round_down=False):
    """
    Given an index and a target index, find the closest index in the index

    Parameters
    ----------
    index
    idx_target
    round_down

    Returns
    -------

    """
    if isinstance(idx_target, list):
        idx_closest = [calc_closest_index(index, idx, return_idx=return_idx) for idx in idx_target]
        return idx_closest
    idx_closest = np.abs(index - idx_target).argmin()
    if round_down and idx_closest > idx_target:
        idx_closest -= 1

    if return_idx:
        return index[idx_closest]
    else:
        return idx_closest


def reindex_with_new_diff(index, diff):
    """
    Given an index and a new difference, return a new index with the same length

    Parameters
    ----------
    index
    diff

    Returns
    -------

    """
    new_index = np.zeros_like(index)
    new_index[0] = index[0]
    for i in range(1, len(index)):
        new_index[i] = new_index[i - 1] + diff
    return new_index
