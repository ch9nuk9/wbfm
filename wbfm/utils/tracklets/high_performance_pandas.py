import logging
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import to_sparse_multiindex, \
    get_contiguous_blocks_from_column
from wbfm.utils.external.utils_neuron_names import name2int_neuron_and_tracklet, int2name_using_mode
from wbfm.utils.tracklets.utils_splitting import TrackletSplitter


class PaddedDataFrame(pd.DataFrame):
    _metadata = ["remaining_empty_column_names", "name_mode", "new_name_generator", "default_num_to_add",
                 "nonempty_column_names"]

    def setup(self, name_mode, initial_empty_cols=100, default_num_to_add=None):
        if default_num_to_add is None:
            self.default_num_to_add = 10000
        else:
            self.default_num_to_add = default_num_to_add
        self.name_mode = name_mode
        new_name_generator = get_next_name_generator(self, name_mode=name_mode)
        self.new_name_generator = new_name_generator
        self.remaining_empty_column_names = []

        if not hasattr(self, 'nonempty_column_names') or self.nonempty_column_names is None:
            self.nonempty_column_names = []

        new_df = self.add_new_empty_column_if_none_left(num_to_add=initial_empty_cols)

        return new_df

    @staticmethod
    def construct_from_basic_dataframe(df, name_mode, initial_empty_cols=100, default_num_to_add=None):
        out = check_if_heterogenous_columns(df)
        if out is not None:
            logging.warning("Padded dataframe will not work as expected when the dataframe has heterogeous columns")
            raise NotImplementedError
        # Note: this does more copying than necessary... could be optimized
        df_pad = PaddedDataFrame(data=df.values, columns=df.columns, index=df.index,
                                 dtype=pd.SparseDtype(float, np.nan))
        # df_pad = PaddedDataFrame(df.astype(pd.SparseDtype(float, np.nan)))
        df_pad.nonempty_column_names = get_names_from_df(df)
        df_pad = df_pad.setup(name_mode=name_mode, initial_empty_cols=initial_empty_cols,
                              default_num_to_add=default_num_to_add)

        return df_pad

    @property
    def num_empty_columns(self):
        return len(self.remaining_empty_column_names)

    def new_like_self(self, df=None):
        if df is None:
            df_padded = PaddedDataFrame(self)
        else:
            df_padded = PaddedDataFrame(df)
        for attr in self._metadata:
            val = self.__getattr__(attr)
            df_padded.__setattr__(attr, val)
        return df_padded

    def drop_empty_columns(self):
        self.drop(columns=self.remaining_empty_column_names, level=0, inplace=True)
        self.remaining_empty_column_names = []

    def return_normal_dataframe(self):
        self.drop_empty_columns()
        return pd.DataFrame(self)

    def return_sparse_dataframe(self):
        self.drop_empty_columns()
        return pd.DataFrame(self).astype(pd.SparseDtype(float, np.nan))

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
        new_cols = self.generate_new_columns(num_to_add)

        # Yikes, two copies... but this should be very rare
        # In theory, the _constructor_expanddim constructor should work here, but it doesn't seem to be
        new_df = self.new_like_self(self.join(new_cols, sort=False))
        print(f"Adding {num_to_add} empty columns; now has {self.num_empty_columns} empty columns")
        return new_df

    def generate_new_columns(self, num_to_add):
        # Add correctly sorted column names
        new_names = [name for _, name in zip(range(num_to_add), self.new_name_generator)]
        self.remaining_empty_column_names.extend(new_names)
        new_cols = empty_dataframe_like(self, new_names)
        return new_cols

    def add_new_empty_column_if_none_left(self, min_empty_cols=1, num_to_add=None):
        if num_to_add is None:
            num_to_add = self.default_num_to_add
        # Weird: the output must be assigned to actually save the new columns
        # ... this is true even though the columns are actually added inplace!
        if self.num_empty_columns < min_empty_cols:
            new_df = self.copy_and_add_empty_columns(num_to_add)
            return new_df
        else:
            return self

    def get_next_empty_column_name(self):
        # This should update in-place, but only if I assign the output
        df = self.add_new_empty_column_if_none_left()
        next_name = df.remaining_empty_column_names.pop(0)
        return next_name

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
        self[right_name] = right_half[right_name]
        self[left_name] = left_half[left_name]
        self.nonempty_column_names.append(right_name)
        return True, self, left_name, right_name

    def split_tracklet_multiple_times(self, split_list, old_name, verbose=1):
        df_working_copy = self.add_new_empty_column_if_none_left(min_empty_cols=len(split_list))
        old_tracklet = df_working_copy[[old_name]]
        new_tracklets = split_multiple_tracklets(old_tracklet, split_list)
        all_new_names = []
        for i, tracklet in enumerate(new_tracklets):
            if i > 0:
                new_name = df_working_copy.get_next_empty_column_name()
            else:
                new_name = old_name
            # They all have the old name
            df_working_copy[new_name] = tracklet[old_name]
            all_new_names.append(new_name)
        self.nonempty_column_names.extend(all_new_names[1:])  # Do not copy the first name, which is the old name

        return df_working_copy, all_new_names

    def split_all_tracklets_using_mode(self, split_mode='gap', verbose=0, DEBUG=False):
        possible_modes = ['gap', 'jump']
        split_mode = split_mode.lower()
        assert split_mode in possible_modes, f"Found split_mode={split_mode}, but it must be one of {possible_modes}"
        if split_mode == 'jump':
            tracklet_splitter = TrackletSplitter(verbose=verbose)

        all_names = get_names_from_df(self)
        name_mapping = defaultdict(set)
        df_working_copy = self

        for original_name in tqdm(all_names):
            name_mapping[original_name].add(original_name)
            if split_mode == 'gap':
                split_list = get_split_points_at_nans_and_delete_singles(df_working_copy, original_name, verbose)
            elif split_mode == 'jump':
                split_list = tracklet_splitter.get_split_points_using_feature_jumps(df_working_copy, original_name)
            else:
                raise NotImplementedError

            split_list_dict = {original_name: split_list}

            # df_working_copy = self.split_using_dict_of_points(name_mapping, split_list_dict)
            # Static method. Note that setting df_working_copy = self means that self should have additional columns too
            df_working_copy, name_mapping = self.split_using_dict_of_points(df_working_copy,
                                                                            split_list_dict, name_mapping)
            if DEBUG:
                break

        return df_working_copy, name_mapping

    @staticmethod
    def split_using_dict_of_points(df_working_copy, split_list_dict, name_mapping=None):
        # Note: this is a static method because I want to explicitly return the dataframe... but I'm not sure if that's
        # necessary or good
        if name_mapping is None:
            name_mapping = defaultdict(set)
        for name, these_splits in tqdm(split_list_dict.items(), leave=False):
            if len(these_splits) >= 1:
                num_to_add = max([5 * len(these_splits), 10000])
                df_working_copy = df_working_copy.add_new_empty_column_if_none_left(
                    min_empty_cols=2 * len(these_splits),
                    num_to_add=num_to_add)
                df_working_copy, all_new_names = df_working_copy.split_tracklet_multiple_times(these_splits, name)
                name_mapping[name].update(all_new_names)
        return df_working_copy, name_mapping


def get_split_points_at_nans_and_delete_singles(df_working_copy, original_name, verbose):
    tracklet = df_working_copy[original_name]['z']
    block_starts, block_ends = get_contiguous_blocks_from_column(tracklet)
    # First delete any isolated ones
    num_deleted = 0
    split_list = []
    for i, (i_start, i_end) in enumerate(zip(block_starts, block_ends)):
        if i_end - i_start == 1:
            # Then it was a length-1 tracklet, so just delete it
            if verbose >= 2:
                print(f"Deleting length-1 tracklet from {original_name}")
            insert_value_in_sparse_df(df_working_copy, i_start, original_name, np.nan)
            num_deleted += 1
        elif i > num_deleted:
            # Don't split if it's the first non-deleted one
            split_list.append(i_start)
            # flag, _, _, right_name = df_working_copy.split_tracklet(i_start, name_of_current_block, verbose=verbose-1)
            # name_mapping[original_name].add(right_name)
            # name_of_current_block = right_name
    return split_list


def insert_value_in_sparse_df(df: pd.DataFrame, index: Union[int, str, list], columns: pd.MultiIndex, val):
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
    if type(columns) == pd.MultiIndex:
        tmp_cols = df[columns].copy()
    else:
        logging.warning(f"Called insert_value_in_sparse_df without specifying the exact index; "
                        "this assumes that val is ordered the same as df, and is not recommended. "
                        f"Ignore this warning if val is nan: {val}")
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
    # df_gt = df_gt.drop(level=1, columns='raw_tracklet_id')

    if gt_names is None:
        # Assume all are correct
        df_gt_just_cols = df_gt.loc[:, (slice(None), col_to_check)]
    else:
        df_gt_just_cols = df_gt.loc[:, (gt_names, col_to_check)]
    if hasattr(df_gt_just_cols, 'sparse'):
        df_gt_just_cols = df_gt_just_cols.sparse.to_dense()
    # Need to speed up, so unpack
    df_just_cols = df_tracker.loc[:, (slice(None), col_to_check)]
    if hasattr(df_gt_just_cols, 'sparse'):
        df_just_cols = df_gt_just_cols.sparse.to_dense()
    ind_to_delete = defaultdict(list)

    t_list = list(range(df_gt.shape[0]))
    if DEBUG:
        t_list = t_list[:5]

    # Get indices (row) to delete (longest step)
    gt_at_all_times = df_gt_just_cols.T.to_dict(orient='list')
    tracks_at_all_times = df_just_cols.T.to_dict(orient='dict')
    # tracklet_names = get_names_from_df(df_just_cols)
    for t in tqdm(t_list):
        gt_at_this_time = set(gt_at_all_times[t])
        values_at_this_time = tracks_at_all_times[t]

        # assert len(values_at_this_time) == len(tracklet_names)
        for tracklet_name_tuple, val in values_at_this_time.items():
            if ~np.isnan(val) and val in gt_at_this_time:
                ind_to_delete[tracklet_name_tuple[0]].append(t)
                # print(tracklet_name, t)
        # gt_at_this_time = set(df_gt_just_cols.loc[t, :])
        # tracks_at_this_time = df_just_cols.loc[t, :]
        #
        # for index, val in tracks_at_this_time.iteritems():
        #     if val in gt_at_this_time:
        #         ind_to_delete[index[0]].append(t)

    # Get corresponding column indices (full zxy, not just index column)
    tracklet_name_to_array_index = defaultdict(list)
    for i, c in enumerate(df_tracker.columns):
        tracklet_name_to_array_index[c[0]].append(i)

    # Actually delete from the array
    values_as_array = df_tracker.values
    for name, t_list in ind_to_delete.items():
        cols = tracklet_name_to_array_index[name]
        # values_as_array[t_list[0]:t_list[-1]+1, cols[0]:cols[-1]+1] = np.nan
        values_as_array[np.array(t_list), cols[0]:cols[-1]+1] = np.nan

    # Recast as pandas
    df_out = pd.DataFrame(data=values_as_array, index=df_tracker.index, columns=df_tracker.columns)
    df_out.dropna(axis=1, how='all', inplace=True)

    return df_out, ind_to_delete, tracklet_name_to_array_index

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


def get_names_from_df(df, level=0, to_sort=True):
    """
    If you do .columns.levels[0] it will not return the update properly!

    Note that not sorting doesn't necessarily preserve the original order
    """
    if type(df) == PaddedDataFrame:
        return df.nonempty_column_names
    names = list(set(df.columns.get_level_values(level)))
    if to_sort:
        names.sort()
    return names


def get_names_of_columns_that_exist_at_t(df, t):
    """Note: MUST assign the output of dropna, otherwise the columns won't update"""
    out = df.iloc[[t]].dropna(axis=1)
    return get_names_from_df(out)


def find_top_level_name_by_single_column_entry(df, t, value, subcolumn_to_check='raw_neuron_ind_in_list'):
    """Assumes a multi-index format, with subcolumn_to_check existing at level 1"""
    df_at_time = df.iloc[[t]].dropna(axis=1)
    df_mask = df_at_time.loc(axis=1)[:, subcolumn_to_check] == value
    df_match = df_mask[df_mask].dropna(axis=1)
    name = get_names_from_df(df_match)

    return name


def empty_dataframe_like(df_tracklets, new_names, dtype=pd.SparseDtype(float, np.nan)):
    # New: force all to be the same columns
    # Initialize using the index and column structure of the tracklets
    cols = pd.MultiIndex.from_product([new_names, df_tracklets.columns.levels[1]])
    vals = np.zeros((df_tracklets.shape[0], len(cols)))
    vals[:] = np.nan

    df_new = pd.DataFrame(data=vals, index=df_tracklets.index, columns=cols, dtype=dtype)

    return df_new


def get_name_mapping_for_track_dataframes(df_old, df_new,
                                          t_templates=10, column_to_test='raw_neuron_ind_in_list',
                                          names_old=None,
                                          verbose=0):
    """Checks for matches in all templates, and takes the most common match"""
    names_new = get_names_from_df(df_new)
    if np.isscalar(t_templates):
        t_templates = [t_templates]
    if names_old is None:
        names_old = get_names_from_df(df_old)
    # dfold2dfnew_dict = defaultdict(int)
    out_dict = {}
    dfold2dfnew_dict = {}
    for old in names_old:
        dfold2dfnew_dict[old] = defaultdict(int)

    # Count matches on all templates
    for t in t_templates:
        for old in tqdm(names_old, leave=False):
            try:
                old_ind = df_old.loc[t, (old, column_to_test)]
            except KeyError:
                # Template time is outside the tracked time
                break
            for new in names_new:
                try:
                    new_ind = df_new.loc[t, (new, column_to_test)]
                except KeyError:
                    # Template time is outside the tracked time
                    break
                if old_ind == new_ind:
                    dfold2dfnew_dict[old][new] += 1
                    # dfold2dfnew_dict[old] = new
                    break
            else:
                if verbose >= 2:
                    print(f"Did not find new neuron for ground truth {old}")

    # Take max
    for old, new_matches in dfold2dfnew_dict.items():
        if new_matches:
            i_best_match = np.argmax(list(new_matches.values()))
            name_best_match = list(new_matches.keys())[i_best_match]
            out_dict[old] = name_best_match

    return out_dict, dfold2dfnew_dict


def plot_pandas_interactive(df):
    from ipywidgets import interact

    names = get_names_from_df(df)
    subcolumns = get_names_from_df(df, level=1)

    def f(name, subcol):
        this_col = df.loc[slice(None), (name, subcol)]
        this_col.plot()
        plt.title(f"Number of non-nan values: {this_col.notnull()}")
        plt.show()

    return interact(f, name=names, subcol=subcolumns)


def check_if_heterogenous_columns(df, verbose=1, raise_error=False):
    names = get_names_from_df(df)
    expected_size = df[names[0]].shape[1]
    for name in names:
        if df[name].shape[1] != expected_size:
            logging.warning(f"Expected {expected_size} columns, but found {df[name].shape[1]}")
            if verbose >= 1:
                print(df[name].columns)
            if raise_error:
                raise NotImplementedError
            return name, df[name]
    else:
        print("No heterogenous columns detected")
    return None


def get_next_name_generator(df, name_mode):
    """See get_next_name_tracklet_or_neuron; this is just a generator version for doing many

    name_mode should be 'tracklet' or 'neuron'; see int2name_using_mode
    """
    all_names = get_names_from_df(df)
    max_int = name2int_neuron_and_tracklet(all_names[-1])
    # Really want to make sure we are after all other names
    i_tracklet = max_int + 1
    i = 0
    while True:
        new_name = int2name_using_mode(i_tracklet + i, name_mode)
        i += 1
        yield new_name


def split_single_sparse_tracklet(i_split, this_tracklet: pd.DataFrame):
    if hasattr(this_tracklet, 'sparse'):
        this_tracklet = this_tracklet.sparse.to_dense()
    else:
        # logging.warning(f"Tracklet {this_tracklet} was already dense")
        pass

    left_half = this_tracklet.copy()
    right_half = this_tracklet.copy()
    left_half.iloc[i_split:] = np.nan
    right_half.iloc[:i_split] = np.nan

    left_half = to_sparse_multiindex(left_half)
    right_half = to_sparse_multiindex(right_half)
    return left_half, right_half


def split_multiple_tracklets(this_tracklet: pd.DataFrame, split_list: list):
    """Splits tracklet into multiple pieces, each with the same name as the original"""
    if hasattr(this_tracklet, 'sparse'):
        this_tracklet = this_tracklet.sparse.to_dense()

    new_tracklets_list = []
    split_list_copy = split_list.copy()
    split_list_copy.sort()
    split_list_copy.insert(0, 0)
    split_list_copy.append(-1)
    for i in range(len(split_list_copy) - 1):
        i_start = split_list_copy[i]
        i_end = split_list_copy[i + 1]
        new_tracklet = this_tracklet.copy()
        if i_start > 0:
            new_tracklet.iloc[:i_start] = np.nan
        if i_end != -1:
            new_tracklet.iloc[i_end:] = np.nan
        new_tracklets_list.append(new_tracklet)

    return new_tracklets_list
