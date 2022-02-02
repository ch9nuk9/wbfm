import numpy as np
import pandas as pd


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
