import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from tqdm.auto import tqdm

from wbfm.utils.general.postprocessing.postprocessing_utils import filter_dataframe_using_likelihood
from wbfm.utils.projects.utils_filenames import read_if_exists, get_sequential_filename
# Note: following must be present, even if pycharm cleans it
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from wbfm.utils.projects.utils_project import safe_cd


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


def scale_impute_descale(df_only_locations: pd.DataFrame, n_nearest_features=20, random_state=0, max_iter=20,
                         estimator=BayesianRidge(), imputer_kwargs=None, to_scale=True):
    if imputer_kwargs is None:
        imputer_kwargs = {}
    all_nan_columns, df_no_all_nan = remove_all_nan_columns(df_only_locations)
    df_dat = df_no_all_nan.to_numpy()

    # This gray import must be present
    from sklearn.impute._iterative import IterativeImputer
    imputer = IterativeImputer(estimator=estimator,
                               random_state=random_state,
                               missing_values=np.nan,
                               verbose=1,
                               n_nearest_features=n_nearest_features,
                               max_iter=max_iter,
                               **imputer_kwargs)
    if to_scale:
        scaler = StandardScaler()
        dat_normalized = scaler.fit_transform(df_dat)
        dat_sklearn = imputer.fit_transform(dat_normalized)
        dat_sklearn = scaler.inverse_transform(dat_sklearn)
    else:
        dat_sklearn = imputer.fit_transform(df_dat)
        scaler = None
    df_sklearn = pd.DataFrame(data=dat_sklearn, columns=df_no_all_nan.columns)

    df_sklearn = replace_all_nan_columns(all_nan_columns, df_sklearn)

    return df_sklearn, imputer, scaler


def replace_all_nan_columns(all_nan_columns, df_sklearn):
    for col in all_nan_columns:
        df_sklearn[col] = np.nan
    return df_sklearn


def remove_all_nan_columns(df_only_locations):
    all_nan_columns = df_only_locations.columns[df_only_locations.count() == 0]
    if len(all_nan_columns) > 0:
        logging.info(f'Some columns are all nan, and are dropped: {all_nan_columns}')
        df_no_all_nan = df_only_locations.dropna(axis=1, how='all')
    else:
        df_no_all_nan = df_only_locations
    return all_nan_columns, df_no_all_nan


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

    return pd.DataFrame(df_interp)  # Reduce fragmentation


def impute_tracks_from_config(tracks_config):

    df_raw, fname, likelihood_thresh, n_nearest_features = _unpack_for_imputing(tracks_config)

    # Preprocessing the multiindex dataframe
    df_filtered = filter_dataframe_using_likelihood(df_raw.copy(), likelihood_thresh)
    df_only_locations, old2new_names = df_of_only_locations(df_filtered)

    # Impute and update
    df_imputed = scale_impute_descale(df_only_locations, n_nearest_features)[0]
    df_final = update_dataframe_using_flat_names(df_raw, df_imputed, old2new_names)

    # Save
    fname = tracks_config.resolve_relative_path_from_config('missing_data_imputed_df')
    abs_fname = get_sequential_filename(fname)
    df_final.to_hdf(abs_fname, key='df_with_missing')

    rel_fname = tracks_config.unresolve_absolute_path(abs_fname)
    logging.info(f"Saving output file at: {rel_fname}")
    tracks_config.config.update({'missing_data_imputed_df': rel_fname})
    tracks_config.update_self_on_disk()


def _unpack_for_imputing(tracks_config):
    fname = tracks_config.resolve_relative_path_from_config('final_3d_tracks_df')
    df_raw = read_if_exists(fname)
    logging.info(f"Reading initial dataframe: {fname}")
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


def get_distance_to_closest_neurons_over_time(project_data, which_neuron, df_track):
    all_dist = []

    for i_frame in tqdm(range(project_data.num_frames)):
        target_pt = df_track[which_neuron].iloc[i_frame][:3]
        all_dist.append(project_data.get_distance_to_closest_neuron(i_frame, target_pt))

    return np.array(all_dist)


## Getting variance
# From: https://github.com/scikit-learn/scikit-learn/blob/e10edc3ddd94f49a48be02c73e68e359b3dad925/examples/impute/plot_multiple_imputation.py
def get_results_chained_imputation(X_ampute, random_state=0, max_iter=10):
    # Impute incomplete data with IterativeImputer using single imputation
    # We perform MAX_ITER imputations and only use the last imputation.

    from sklearn.impute._iterative import IterativeImputer
    imputer = IterativeImputer(max_iter=max_iter,
                               sample_posterior=True,
                               random_state=random_state)
    X_imputed = imputer.fit_transform(X_ampute)
    # For me, just return the data without fitting a predictive model
    return X_imputed


def estimates_and_variance_mice_imputation(X_ampute, n_imputations=5, max_iter=10):
    # Fill the data multiple times (different random seeds)
    f = lambda i: get_results_chained_imputation(X_ampute, random_state=i, max_iter=max_iter)
    multiple_datasets = [f(i) for i in tqdm(list(range(n_imputations)))]

    point_estimates = np.mean(multiple_datasets, axis=0)
    var_estimates = np.var(multiple_datasets, axis=0)

    return point_estimates, var_estimates


def impute_missing_values_in_dataframe(df: pd.DataFrame, d=None) -> pd.DataFrame:
    """
    Given a dataframe with gaps, impute the missing values using PPCA

    Parameters
    ----------
    df
    d

    Returns
    -------

    """
    from ppca import PPCA

    # DLC uses zeros as "failed tracking"
    # Replace with nan and scale
    df.replace(0, np.NaN, inplace=True)
    df_dat = df.to_numpy()
    scaler = StandardScaler()
    scaler.fit(df_dat)
    dat_normalized = scaler.transform(df_dat)
    # Actually impute
    ppca = PPCA()
    ppca.fit(data=dat_normalized, d=d, verbose=False)
    dat_imputed = scaler.inverse_transform(ppca.data)
    df_imputed = pd.DataFrame(data=dat_imputed, columns=df.columns)
    return df_imputed


def impute_missing_values_using_config(tracking_config, DEBUG=False):
    """
    Using gappy time series of the positions of all neurons, uses probabilistic PCA (PPCA) to impute the missing values


    Parameters
    ----------
    project_config

    Returns
    -------

    """

    # Unpack config
    project_dir = tracking_config['project_dir']
    df_fname = tracking_config['final_3d_tracks_df']

    #
    with safe_cd(project_dir):
        df = pd.read_hdf(df_fname)

    df_imputed = impute_missing_values_in_dataframe(df)

    # Save
    out_fname = Path(df_fname).with_name('postprocessing').joinpath('full_3d_tracks.h5')
    df_imputed.to_hdf(out_fname, 'df_with_missing')
    df_imputed.to_csv(Path(out_fname).with_suffix('.csv'))
