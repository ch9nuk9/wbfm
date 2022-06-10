import logging
from collections import defaultdict
import numpy as np
import pandas as pd


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


def get_times_of_conflicting_dataframes(tracklet_list: list,
                                        tracklet_network_names: list,
                                        verbose=0):
    """
    Takes a list of tracklets and their names, and finds the conflict points between the tracklets in the list

    Returns the times of the conflict points

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
    for i1, (idx1, base_tracklet_name) in enumerate(zip(all_indices, tracklet_network_names)):
        if len(idx1) == 0:
            logging.warning(f"Skipping empty tracklet {base_tracklet_name}")
            continue
        idx1_edges = [int(idx1[0]), int(idx1[-1])+1]
        for idx2, target_tracklet_name in zip(all_indices[i1 + 1:], tracklet_network_names[i1 + 1:]):
            if base_tracklet_name == target_tracklet_name:
                logging.warning("Attempted to compare tracklet to itself")
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


def cast_int_or_nan(i):
    """Cast as integer, but do not crash if np.nan"""
    if np.isnan(i):
        return i
    else:
        return int(i)


def get_contiguous_blocks_from_column(tracklet: pd.Series) -> Tuple[list, list]:
    """
    Given a pd.Series that may have gaps, get the indices of the contiguous blocks of non-nan points

    Parameters
    ----------
    tracklet

    Returns
    -------

    """
    if hasattr(tracklet, 'sparse'):
        change_ind = np.where(tracklet.isnull().sparse.to_dense().diff().values)[0]
    else:
        change_ind = np.where(tracklet.isnull().diff().values)[0]
    block_starts = []
    block_ends = []

    for i in change_ind:
        if np.isnan(tracklet[i]):
            if i > 0:
                # Diff always has a value here, but it can only be a start, not an end
                block_ends.append(i)
        else:
            block_starts.append(i)
    return block_starts, block_ends


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
    fn = 0
    unknown = 0
    gt_dict = dict(gt_matches)
    for m in new_matches:
        gt_val = gt_dict.get(m[0], None)
        if allow_unknown and gt_val is None:
            unknown += 1
        elif null_value in m:
            fn += 1
        elif m[1] == gt_val:
            tp += 1
        else:
            fp += 1
    return tp, fp, fn, unknown


def fill_missing_indices_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe that may skip time points (e.g. the Index is 1, 2, 5), fill the missing Index values with nan

    Parameters
    ----------
    df

    Returns
    -------

    """
    t = df.index
    if len(t) != int(t[-1]) + 1:
        add_indices = pd.Index(range(int(t[-1]))).difference(t)
        add_df = pd.DataFrame(index=add_indices, columns=df.columns)
        df = pd.concat([df, add_df]).sort_index()
        num_added = add_df.shape[0]
    else:
        num_added = 0
    return df, num_added


def get_column_name_from_time_and_column_value(df, i_time, col_value, col_name):
    mask = df.loc[i_time, (slice(None), col_name)] == col_value
    try:
        ind = np.where(mask)[0][0]
        return ind, mask.index.levels[0][mask][0]
    except IndexError:
        return None, None
