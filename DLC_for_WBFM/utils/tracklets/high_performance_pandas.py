import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_pandas import get_names_from_df, empty_dataframe_like, to_sparse_multiindex
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name_using_mode, name2int_neuron_and_tracklet, \
    int2name_dummy
from DLC_for_WBFM.utils.tracklets.utils_tracklets import split_single_tracklet


class PaddedDataFrame(pd.DataFrame):

    _metadata = ["dummy_name_template", "remaining_dummy_names"]

    @property
    def _constructor(self):
        return PaddedDataFrame

    @property
    def num_dummy_columns(self):
        return len(self.remaining_dummy_names)

    def add_dummy_columns(self, num_to_add=10):
        # Must be correct type

        i_start = self.num_dummy_columns
        new_names = [int2name_dummy(i_start + i) for i in range(num_to_add)]
        self.remaining_dummy_names.extend(new_names)

        print(self.shape)
        new_cols = empty_dataframe_like(self, new_names)
        # print(new_cols)

        df_new = self.join(new_cols, sort=False)
        print(df_new.shape)

        df_new.remaining_dummy_names = self.remaining_dummy_names
        return df_new

    def get_next_name_tracklet_or_neuron(self, name_mode="tracklet"):
        all_names = get_names_from_df(self)
        all_names = [n for n in all_names if "zzz" not in n]
        max_int = name2int_neuron_and_tracklet(all_names[-1])
        # Really want to make sure we are after all other names
        i_tracklet = max_int + 1
        new_name = int2name_using_mode(i_tracklet, name_mode)
        assert new_name not in all_names, "Failed to generate new tracklet name"
        return new_name

    def get_next_dummy_name(self):
        return self.remaining_dummy_names.pop(-1)


def split_tracklet_within_padded_dataframe(all_tracklets, i_split, old_name, verbose=1):
    left_name = old_name
    this_tracklet = all_tracklets[[left_name]]
    idx = this_tracklet.index[this_tracklet[left_name]['z'].notnull()]
    # if i_split not in this_tracklet.dropna(axis=0).index:
    if i_split not in idx:
        logging.warning(f"Tried to split {old_name} at {i_split}, but it doesn't exist at that time")
        return False, all_tracklets, left_name, None
    # Split
    left_half, right_half = split_single_tracklet(i_split, this_tracklet)
    right_name = all_tracklets.get_next_name_tracklet_or_neuron()
    right_half.rename(columns={left_name: right_name}, level=0, inplace=True)
    if verbose >= 1:
        print(f"Creating new tracklet {right_name} from {left_name} by splitting at t={i_split}")
        print(
            f"New non-nan lengths: new: {right_half[right_name]['z'].count()}, old:{left_half[left_name]['z'].count()}")
        # Performance tests
    dummy_name = all_tracklets.get_next_dummy_name()
    all_tracklets[dummy_name] = right_half[right_name]
    all_tracklets[left_name] = left_half[left_name]
    all_tracklets.rename(columns={dummy_name: right_name}, level=0, inplace=True)
    return True, all_tracklets, left_name, right_name


def insert_value_in_sparse_df(df, index, columns, val):
    """ Insert data in a DataFrame with SparseDtype format

    from: https://stackoverflow.com/questions/49032856/assign-values-to-sparsearray-in-pandas

    Only applicable for pandas version > 0.25

    Args
    ----
    df : DataFrame with series formatted with pd.SparseDtype
    index: str, or list, or slice object
        Same as one would use as first argument of .loc[]
    columns: str, list, or slice
        Same one would normally use as second argument of .loc[]
    val: insert values

    Returns
    -------
    df: DataFrame
        Modified DataFrame

    """
    # Save the original sparse format for reuse later
    spdtypes = df.dtypes[columns]

    # Convert concerned Series to dense format, but MAKE SURE it is actually sparse!
    tmp_cols = df[[columns]].copy()
    try:
        tmp_cols = tmp_cols.sparse.to_dense()
    except AttributeError:
        # Then it should already be dense
        # tmp_cols = to_sparse_multiindex(tmp_cols)
        pass
    # df[columns] = df[columns].sparse.to_dense()

    # Do a normal insertion with .loc[]
    tmp_cols.loc[index, columns] = val

    # Back to the original sparse format
    # NOTE: multiindex assignment must be done per-column, otherwise it reverts to dense
    df = to_sparse_multiindex(df, tmp_cols)

    return df


def delete_tracklets_using_ground_truth(df_gt, df_tracker, col='raw_neuron_ind_in_list'):
    """Loops through both ground truth and tracklets, and deletes any tracklets that conflict with the gt"""
    gt_names = get_names_from_df(df_gt)
    tracklet_names = get_names_from_df(df_tracker)

    for gt_name in tqdm(gt_names):
        gt_col = df_gt[gt_name, col]

        for tracklet_name in tqdm(tracklet_names, leave=False):
            overlap = df_tracker[tracklet_name, col] == gt_col
            if any(overlap):
                t = np.where(overlap)[0]
                df_tracker = insert_value_in_sparse_df(df_tracker, t, tracklet_name, np.nan)

    return df_tracker


def dataframe_equal_including_nan(df, df2):
    # From: https://stackoverflow.com/questions/19322506/pandas-dataframes-with-nans-equality-comparison
    return (df == df2) | ((df != df) & (df2 != df2))
