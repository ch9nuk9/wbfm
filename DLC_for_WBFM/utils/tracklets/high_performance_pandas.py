import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_pandas import empty_dataframe_like, to_sparse_multiindex, get_names_from_df, \
    get_contiguous_blocks_from_column, check_if_heterogenous_columns
from DLC_for_WBFM.utils.tracklets.utils_tracklets import split_single_sparse_tracklet, get_next_name_generator


class PaddedDataFrame(pd.DataFrame):
    _metadata = ["remaining_empty_column_names", "name_mode", "new_name_generator"]

    def setup(self, name_mode, initial_empty_cols=100):
        self.name_mode = name_mode
        new_name_generator = get_next_name_generator(self, name_mode=name_mode)
        self.new_name_generator = new_name_generator
        self.remaining_empty_column_names = []

        new_df = self.add_new_empty_column_if_none_left(num_to_add=initial_empty_cols)
        return new_df

    @staticmethod
    def construct_from_basic_dataframe(df, name_mode, initial_empty_cols=100):
        out = check_if_heterogenous_columns(df)
        if out is not None:
            logging.warning("Padded dataframe will not work as expected when the dataframe has heterogeous columns")
            raise NotImplementedError
        # Note: this does more copying than necessary... could be optimized
        df_pad = PaddedDataFrame(data=df.values, columns=df.columns, index=df.index,
                                 dtype=pd.SparseDtype(float, np.nan))
        # df_pad = PaddedDataFrame(df.astype(pd.SparseDtype(float, np.nan)))
        df_pad = df_pad.setup(name_mode=name_mode, initial_empty_cols=initial_empty_cols)
        return df_pad

    @property
    def num_empty_columns(self):
        return len(self.remaining_empty_column_names)

    def new_like_self(self, df):
        df_padded = PaddedDataFrame(df)
        for attr in self._metadata:
            val = self.__getattr__(attr)
            df_padded.__setattr__(attr, val)
        return df_padded

    def drop_empty_columns(self):
        self.drop(columns=self.remaining_empty_column_names, inplace=True)
        self.remaining_empty_column_names = []

    def return_normal_dataframe(self):
        self.drop_empty_columns()
        return pd.DataFrame(self)

    @property
    def _constructor(self):
        return PaddedDataFrame

    @property
    def _constructor_expanddim(self):
        return PaddedDataFrame

    # @property
    # Series NOT implemented
    # def _constructor_sliced(self):
    #     return PaddedDataFrame

    def copy_and_add_empty_columns(self, num_to_add):
        print(f"Adding {num_to_add} empty columns")
        new_cols = self.generate_new_columns(num_to_add)

        # Yikes, two copies... but this should be very rare
        # In theory, the _constructor_expanddim constructor should work here, but it doesn't seem to be
        return self.new_like_self(self.join(new_cols, sort=False))

    def generate_new_columns(self, num_to_add):
        # Add correctly sorted column names
        new_names = [name for _, name in zip(range(num_to_add), self.new_name_generator)]
        self.remaining_empty_column_names.extend(new_names)
        # Note: can't add more than double number of columns
        new_cols = empty_dataframe_like(self, new_names)
        return new_cols

    def add_new_empty_column_if_none_left(self, min_empty_cols=1, num_to_add=500):
        if self.num_empty_columns < min_empty_cols:
            new_df = self.copy_and_add_empty_columns(num_to_add)
            return new_df
        else:
            return self

    def get_next_empty_column_name(self):
        return self.remaining_empty_column_names.pop(0)

    def split_tracklet(self, i_split, old_name, verbose=1):
        left_name = old_name
        this_tracklet = self[[left_name]]
        idx = this_tracklet.index[this_tracklet[left_name]['z'].notnull()]
        if i_split not in idx:
            logging.warning(f"Tried to split {old_name} at {i_split}, but it doesn't exist at that time")
            return False, self, left_name, None
        # Split
        left_half, right_half = split_single_sparse_tracklet(i_split, this_tracklet)
        right_name = self.get_next_empty_column_name()
        right_half.rename(columns={left_name: right_name}, level=0, inplace=True)
        if verbose >= 1:
            print(f"Creating new tracklet {right_name} from {left_name} by splitting at t={i_split}")
            print(
                f"New non-nan lengths: new: {right_half[right_name]['z'].count()}, old:{left_half[left_name]['z'].count()}")
        # new_name = self.get_next_empty_column_name()
        self[right_name] = right_half[right_name]
        self[left_name] = left_half[left_name]
        # Do not rename; assume it was correct
        # self.rename(columns={dummy_name: right_name}, level=0, inplace=True)
        return True, self, left_name, right_name

    def split_all_non_contiguous_tracklets(self, verbose=0):
        all_names = get_names_from_df(self)
        name_mapping = defaultdict(set)

        df_working_copy = self

        for original_name in tqdm(all_names):
            df_working_copy = df_working_copy.add_new_empty_column_if_none_left()

            tracklet = df_working_copy[original_name]['z']
            block_starts, block_ends = get_contiguous_blocks_from_column(tracklet)
            name_mapping[original_name].add(original_name)
            # First delete any isolated ones
            num_deleted = 0
            name_of_current_block = original_name
            for i, (i_start, i_end) in enumerate(zip(block_starts, block_ends)):
                if i_end - i_start == 1:
                    # Then it was a length-1 tracklet, so just delete it
                    if verbose >= 2:
                        print(f"Deleting length-1 tracklet from {name_of_current_block}")
                    insert_value_in_sparse_df(df_working_copy, i_start, name_of_current_block, np.nan)
                    num_deleted += 1
                elif i > num_deleted:
                    # Don't split if it's the first non-deleted one
                    flag, _, _, right_name = df_working_copy.split_tracklet(i_start, name_of_current_block, verbose=verbose-1)
                    name_mapping[original_name].add(right_name)
                    name_of_current_block = right_name

        return df_working_copy, name_mapping


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
    # Convert concerned Series to dense format, but MAKE SURE it is actually sparse!
    tmp_cols = df[[columns]].copy()
    try:
        tmp_cols = tmp_cols.sparse.to_dense()
    except AttributeError:
        # Then it should already be dense
        # tmp_cols = to_sparse_multiindex(tmp_cols)
        pass

    # Do a normal insertion with .loc[]
    tmp_cols.loc[index, columns] = val

    # Back to the original sparse format
    # NOTE: multiindex assignment must be done per-column, otherwise it reverts to dense
    df = to_sparse_multiindex(df, tmp_cols)

    return df


def delete_tracklets_using_ground_truth(df_gt, df_tracker, gt_names=None,
                                        col_to_check='raw_neuron_ind_in_list', DEBUG=False):
    """Loops through both ground truth and tracklets, and deletes any tracklets that conflict with the gt"""
    # Remove extra column added in steps after the tracklets
    df_gt = df_gt.drop(level=1, columns='raw_tracklet_id')

    if gt_names is None:
        # Assume all are correct
        df_gt_just_cols = df_gt.loc[:, (slice(None), col_to_check)]
    else:
        df_gt_just_cols = df_gt.loc[:, (gt_names, col_to_check)]
    # Need to speed up, so unpack
    df_just_cols = df_tracker.loc[:, (slice(None), col_to_check)]
    ind_to_delete = defaultdict(list)

    t_list = list(range(df_gt.shape[0]))
    if DEBUG:
        t_list = t_list[:5]

    # Get indices (row) to delete (longest step)
    for t in tqdm(t_list):
        gt_at_this_time = set(df_gt_just_cols.loc[t, :])
        tracks_at_this_time = df_just_cols.loc[t, :]

        for index, val in tracks_at_this_time.iteritems():
            if val in gt_at_this_time:
                ind_to_delete[index[0]].append(t)

    # Get corresponding column indices (full zxy, not just index column)
    tracklet_name_to_array_index = defaultdict(list)
    for i, c in enumerate(df_tracker.columns):
        tracklet_name_to_array_index[c[0]].append(i)

    # Actually delete from the array
    values_as_array = df_tracker.values
    for name, t_list in ind_to_delete.items():
        cols = tracklet_name_to_array_index[name]
        values_as_array[t_list[0]:t_list[-1]+1, cols[0]:cols[-1]+1] = np.nan

    # Recast as pandas
    df_out = pd.DataFrame(values_as_array, columns=df_tracker.columns)
    df_out.dropna(axis=1, how='all', inplace=True)

    return df_out

    # tracklet_cols = df_tracker.loc(axis=1)[:, col]
    #
    # for gt_name in tqdm(gt_names):
    #     gt_col = set(df_gt[gt_name, col].astype(int))
    #
    #     for tracklet_name, tracklet_col in tqdm(zip(tracklet_names, df_just_cols), leave=False):
    #         overlap = tracklet_col == gt_col
    #         if any(overlap):
    #             t = np.where(overlap)[0]
    #             if not dry_run:
    #                 df_tracker = insert_value_in_sparse_df(df_tracker, t, tracklet_name, np.nan)

    # return df_tracker


def dataframe_equal_including_nan(df, df2):
    # From: https://stackoverflow.com/questions/19322506/pandas-dataframes-with-nans-equality-comparison
    return (df == df2) | ((df != df) & (df2 != df2))
