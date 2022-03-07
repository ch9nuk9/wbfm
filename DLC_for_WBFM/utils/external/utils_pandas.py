import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def dataframe_to_dataframe_zxy_format(df_tracklets, flip_xy=False) -> pd.DataFrame:
    """Currently, flipxy is true when calling from napari"""
    if not flip_xy:
        coords = ['z', 'x', 'y']
    else:
        coords = ['z', 'y', 'x']
    df_tracklets = df_tracklets.loc(axis=1)[:, coords]
    df_tracklets = df_tracklets.sort_index(axis=1, level=0, sort_remaining=False)
    return df_tracklets


def dataframe_to_numpy_zxy_single_frame(df_tracklets, t, flip_xy=False) -> np.ndarray:
    df_zxy = dataframe_to_dataframe_zxy_format(df_tracklets.iloc[[t], :], flip_xy)
    return df_zxy.to_numpy().reshape(-1, 3)


def get_names_from_df(df, level=0):
    """If you do .columns.levels[0] it will not return the update properly!"""
    names = list(set(df.columns.get_level_values(level)))
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


def get_names_of_conflicting_dataframes(tracklet_list, tracklet_network_names):
    all_indices = [t.dropna().index for t in tracklet_list]
    overlapping_tracklet_ind = []
    overlapping_tracklet_names = []
    for i1, idx1 in enumerate(all_indices):
        this_overlapping_ind = [i1]
        for i2, idx2 in enumerate(all_indices[i1 + 1:]):
            if len(idx1.intersection(idx2)) > 0:
                this_overlapping_ind.append(i2 + i1 + 1)
        if len(this_overlapping_ind) > 1:
            overlapping_tracklet_ind.append(this_overlapping_ind)
            these_names = [tracklet_network_names[n] for n in this_overlapping_ind]
            overlapping_tracklet_names.append(these_names)
    return overlapping_tracklet_names


def empty_dataframe_like(df_tracklets, new_names) -> pd.DataFrame:
    # Initialize using the index and column structure of the tracklets
    all_tracklet_names = get_names_from_df(df_tracklets)
    num_neurons = len(new_names)
    new_names.sort()
    tmp_names = all_tracklet_names[:num_neurons]

    df_new = df_tracklets.loc[:, tmp_names].copy()
    name_mapper = {t: n for t, n in zip(tmp_names, new_names)}
    df_new.rename(columns=name_mapper, inplace=True)
    df_new[:] = np.nan
    return df_new


def check_if_fully_sparse(df):
    # No good way: https://github.com/pandas-dev/pandas/issues/26706
    return df.dtypes.apply(pd.api.types.is_sparse).all()


def to_sparse_multiindex(df, new_columns=None):
    # Must be done in a loop, per column (note: column index will generally be a tuple)
    if new_columns is None:
        new_columns = df
    new_columns = new_columns.astype(pd.SparseDtype("float", np.nan))  # This works, but then direct assignment doesn't
    for c in new_columns.columns:
        df[c] = new_columns[c]

    return df


def cast_int_or_nan(i):
    if np.isnan(i):
        return i
    else:
        return int(i)


def get_name_mapping_for_track_dataframes(df_new, df_old, t_template=10, column_to_test='raw_neuron_ind_in_list'):
    names_new = get_names_from_df(df_new)
    names_old = get_names_from_df(df_old)
    dfold2dfnew_dict = {}

    for old in tqdm(names_old):
        old_ind = df_old.loc[t_template, (old, column_to_test)]
        for new in names_new:
            new_ind = df_new.loc[t_template, (new, column_to_test)]
            if old_ind == new_ind:
                dfold2dfnew_dict[old] = new
    return dfold2dfnew_dict
