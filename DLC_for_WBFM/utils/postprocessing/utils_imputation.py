from pathlib import Path

import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import filter_dataframe_using_likelihood
from DLC_for_WBFM.utils.projects.utils_filepaths import SubfolderConfigFile, read_if_exists
# Note: following must be present, even if pycharm cleans it
# from sklearn.experimental import enable_iterative_imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


def df_of_only_locations(df_raw):
    """
    Convert multi-index dataframe to numpy-ready flat form

    Parameters
    ----------
    df_raw

    Returns
    -------

    """
    old_names = df_raw.columns.to_flat_index()
    old2new_names = {a: "_".join(a) for a in old_names}
    new_names = [old2new_names[n] for n in old_names]
    df_raw.columns = new_names

    to_keep = [c for c in df_raw.columns if is_spatial_column_name(c)]
    df_only_locations = df_raw[to_keep].copy()

    return df_only_locations, old2new_names


def is_spatial_column_name(c):
    return ('likelihood' not in c.lower()) and ('intensity' not in c.lower()) and ('label' not in c.lower()) and \
                ('area' not in c.lower())


def scale_impute_descale(df_only_locations: pd.DataFrame, n_nearest_features=20, random_state=0):
    df_dat = df_only_locations.to_numpy()

    imputer = IterativeImputer(random_state=random_state, missing_values=np.nan, verbose=1,
                               n_nearest_features=n_nearest_features,
                               max_iter=20)
    scaler = StandardScaler()
    scaler.fit(df_dat)
    dat_normalized = scaler.transform(df_dat)

    imputer.fit(dat_normalized)

    dat_sklearn = scaler.inverse_transform(imputer.transform(dat_normalized))
    df_sklearn = pd.DataFrame(data=dat_sklearn, columns=df_only_locations.columns)

    return df_sklearn


def update_dataframe_using_flat_names(df_old, df_new, old2new_names):
    """
    Note: can also be used when only updating part of the column (e.g. with takens embedding)

    Also leaves non-spatial columns as they are
    """

    df_interp = df_old.copy()
    old_names = list(df_old.columns)

    for n in old_names:
        if not is_spatial_column_name(n[1]):
            continue
        df_interp[n] = df_new[old2new_names[n]]

    return df_interp


def impute_tracks_from_config(tracks_config: SubfolderConfigFile):

    df_raw, fname, likelihood_thresh, n_nearest_features = _unpack_for_imputing(tracks_config)

    # Preprocessing the multiindex dataframe
    df_filtered = filter_dataframe_using_likelihood(df_raw.copy(), likelihood_thresh)
    df_only_locations, old2new_names = df_of_only_locations(df_filtered)

    # Impute and update
    df_imputed = scale_impute_descale(df_only_locations, n_nearest_features)
    df_final = update_dataframe_using_flat_names(df_raw, df_imputed, old2new_names)

    # Save
    fname = Path(fname)
    base_fname = fname.stem
    out_fname = fname.with_name(f"{base_fname}_imputed.h5")
    # out_fname = fname.with_stem(f"{base_fname}_imputed")
    df_final.to_hdf(out_fname, key='df_with_missing')


def _unpack_for_imputing(tracks_config):
    fname = tracks_config.resolve_relative_path_from_config('final_3d_tracks_df')
    df_raw = read_if_exists(fname)
    likelihood_thresh = tracks_config.config['missing_data_postprocessing']['likelihood_threshold']
    n_nearest_features = tracks_config.config['missing_data_postprocessing']['n_nearest_features']
    return df_raw, fname, likelihood_thresh, n_nearest_features


##
## For adding time (currently not very stable numerically)
##

def takens_embedding(data, dimension, delay=1, append_dim=0):
    """
    This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)

    The first n columns start at index 0 and end early; the last n columns end at the final index
    """
    # From: https://www.kaggle.com/tigurius/introduction-to-taken-s-embedding
    if delay * dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')

    def get_final_index(i):
        return len(data) - delay * (dimension - i) + 1

    embedded_data = np.array([data[0:get_final_index(0)]])
    for i in range(1, dimension):
        embedded_data = np.append(embedded_data, [data[i * delay:get_final_index(i)]], axis=append_dim)
    return embedded_data