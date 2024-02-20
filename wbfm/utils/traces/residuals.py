from types import SimpleNamespace
from typing import Tuple

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def calculate_residual_subtract_pca(df: pd.DataFrame, n_components=2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Note: must not contain nan"""
    pca = PCA(n_components=n_components, whiten=False)

    df_dat = df.to_numpy()
    scaler = StandardScaler(with_std=False)
    scaler.fit(df_dat)
    dat_normalized = scaler.transform(df_dat)
    dat_low_dimensional = pca.fit_transform(dat_normalized)
    dat_reprojected = pca.inverse_transform(dat_low_dimensional)
    dat_reconstructed = scaler.inverse_transform(dat_reprojected)

    dat_residual = df_dat - dat_reconstructed
    df_residual = pd.DataFrame(data=dat_residual, columns=df.columns, index=df.index)
    df_reconstructed = pd.DataFrame(data=dat_reconstructed, columns=df.columns, index=df.index)

    return df_residual, df_reconstructed


def calculate_residual_subtract_nmf(df: pd.DataFrame, n_components=2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Note: must not contain nan"""
    model = NMF(n_components=n_components)

    df_dat = df.to_numpy()
    all_mins = df_dat.min()
    dat_normalized = df_dat - all_mins
    dat_low_dimensional = model.fit_transform(dat_normalized)
    dat_reconstructed = model.inverse_transform(dat_low_dimensional)
    dat_reconstructed += all_mins

    dat_residual = df - dat_reconstructed
    df_residual = pd.DataFrame(data=dat_residual, columns=df.columns)
    df_reconstructed = pd.DataFrame(data=dat_reconstructed, columns=df.columns)

    return df_residual, df_reconstructed


#
# Analysis
#

def rectified_linear_model_on_all_datasets(df_all_traces, df_all_pca, df_all_behavior,
                                           beh_feature='vb02_curvature', y_name='VB02'):
    """
    Run a rectified linear model on all datasets

    Assumes three dataframes that can be grouped by 'dataset_name', and merged on 'dataset_name' and 'local_time'

    Parameters
    ----------
    df_all_traces
    df_all_behavior

    Returns
    -------

    """
    # Drop any columns that are not real neurons, i.e. contain 'neuron' in the name
    df_all_traces = df_all_traces.loc[:, ~df_all_traces.columns.str.contains('neuron')]
    # Make sure that the local_time column exists and is an integer
    df_all_traces['local_time'] = df_all_traces.groupby('dataset_name').cumcount()
    df_all_pca['local_time'] = df_all_pca.groupby('dataset_name').cumcount()
    df_all_behavior['local_time'] = df_all_behavior.groupby('dataset_name').cumcount()

    # Merge all dataframes
    df_all = df_all_traces.merge(df_all_pca, on=['dataset_name', 'local_time'], how='inner')
    df_all = df_all.merge(df_all_behavior, on=['dataset_name', 'local_time'], how='inner')

    # Define rectified behavior features
    df_all['fwd_curvature'] = df_all[beh_feature] * df_all['fwd'].astype(int)
    df_all['rev_curvature'] = df_all[beh_feature] * (1-df_all['fwd'].astype(int))

    # Loop through all datasets and fit a model to each one
    df_all_models = df_all.groupby('dataset_name').apply(lambda x: rectified_linear_model(x, y_name))
    df_all_models = pd.DataFrame(df_all_models)

    for beh_name in ['rev', 'fwd']:
        df_all_models[f'{y_name}_{beh_name}_pvalue'] = df_all_models[0].apply(
            lambda x: x.pvalues[f'{beh_name}_curvature'])
        df_all_models[f'{y_name}_{beh_name}'] = df_all_models[0].apply(
            lambda x: x.params[f'{beh_name}_curvature'])

    # Plot
    fig = px.box(df_all_models, y=['rev', 'fwd'])
    fig.show()

    return df_all, df_all_models, fig


def rectified_linear_model(df_subset, y_name):
    # Make sure there the neuron exists for this subset
    if y_name in df_subset.dropna(axis=1).columns:
        # assert df_subset.isna().sum().sum() == 0
        return smf.ols(f"{y_name} ~ rev_curvature + fwd_curvature + pca0 + pca1", df_subset).fit()
    else:
        # Make a dummy model
        dummy_dict = SimpleNamespace()
        dummy_dict.params = dict()
        dummy_dict.pvalues = dict()
        for beh_name in ['rev', 'fwd']:
            dummy_dict.params.update({f'{beh_name}_curvature': np.nan})
            dummy_dict.pvalues.update({f'{beh_name}_curvature': np.nan})
        return dummy_dict
