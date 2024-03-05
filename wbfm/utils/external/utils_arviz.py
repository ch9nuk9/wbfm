import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import arviz as az
import pymc as pm

from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir


def plot_ts(idata, y='y', y_hat='y', num_samples=100):
    """
    My implementation of the plot_ts function from ArviZ, which doesn't work for me.

    Uses plotly

    Parameters
    ----------
    idata

    Returns
    -------

    """

    # Build everything in a big dataframe, which will be used to plot everything
    y_obs = idata.observed_data.y

    # Get a sample from the large matrix of predictions
    # Original shape: (num_chains, num_draws, num_timesteps)
    # Target shape: (num_samples, num_timesteps)
    # First flatten the first two dimensions
    try:
        y_hat = idata.posterior_predictive.y
    except AttributeError:
        y_hat = idata.prior_predictive.y
    y_hat = np.array(y_hat).reshape(-1, y_hat.shape[2])
    # Take a random sample from both num_chains and num_draws
    idx = np.random.choice(y_hat.shape[0], num_samples)
    y_hat = y_hat[idx]

    # Combine everything into a single dataframe
    df = pd.DataFrame(y_hat.T)
    df['observed'] = y_obs

    # Plot
    fig = go.Figure()
    # Average of the samples
    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, :-1].mean(axis=1), mode='lines', name='y_hat',
                             line=dict(color='gray', width=2)))
    # 95% CI
    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, :-1].quantile(0.025, axis=1), mode='lines', name='y_hat',
                             line=dict(color='gray', width=1), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, :-1].quantile(0.975, axis=1), mode='lines', name='y_hat',
                             line=dict(color='gray', width=1), fill='tonexty',
                             showlegend=False))

    # Observed
    fig.add_trace(go.Scatter(x=df.index, y=df['observed'], mode='lines', name='observed',
                             line=dict(color='black', width=2)))

    fig.show()
    return fig


def plot_model_elements(idata, y_list=None):
    """

    Parameters
    ----------
    idata
    y_list

    Returns
    -------

    """
    if y_list is None:
        y_list = ['y', 'sigmoid_term', 'curvature_term', 'mu']
    all_pred = {}
    for y_name in y_list:
        if 'posterior_predictive' in idata:
            y_pred = np.mean(np.mean(idata.posterior_predictive[y_name], axis=0), axis=0)
        elif y_name in idata.prior_predictive:
            y_pred = np.mean(np.mean(idata.prior_predictive[y_name], axis=0), axis=0)
        elif y_name in idata.prior:
            y_pred = np.mean(np.mean(idata.prior[y_name], axis=0), axis=0)
        else:
            raise ValueError(f"Could not find {y_name} in idata")
        all_pred[y_name] = y_pred
    df_pred = pd.DataFrame(all_pred)
    # Also get the observed data
    y_obs = idata.observed_data.y
    df_pred['observed'] = y_obs

    fig = px.line(df_pred)
    fig.show()
    return fig


def load_from_disk_and_plot(trace_fname, check_if_exists=True):
    """
    Loads a trace from disk, build the posterior predictive and plot it

    Returns
    -------

    """
    out_fname = trace_fname.replace('.nc', '.html')
    if check_if_exists and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping")
        return

    # Load the trace
    trace = az.from_netcdf(trace_fname)

    # Get the neuron name from the filename
    # Assume it is like */{neuron_name}_{model_name}_trace.nc
    neuron_name = Path(trace_fname).name.split('_')[0]

    if 'posterior_predictive' not in trace:
        print(f"Posterior predictive not found in {trace_fname}. Rebuilding the model")

        # Check if gcamp or gfp
        if Path(trace_fname).is_relative_to(get_hierarchical_modeling_dir()):
            data_fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
        elif Path(trace_fname).is_relative_to(get_hierarchical_modeling_dir(gfp=True)):
            data_fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
        else:
            raise ValueError(f"Could not find the original data file for {trace_fname}")
        Xy = pd.read_hdf(data_fname)
        # Rebuild the model, which is required to build the posterior predictive
        from wbfm.utils.external.utils_pymc import get_dataframe_for_single_neuron
        df_model = get_dataframe_for_single_neuron(Xy, neuron_name)
        x = df_model['x'].values
        y = df_model['y'].values
        curvature = df_model[['eigenworm0', 'eigenworm1', 'eigenworm2']].values

        dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
        coords = {'dataset_name': dataset_name_values}
        dims = 'dataset_name'

        dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)
        with pm.Model(coords=coords) as hierarchical_model:
            # Full model
            from wbfm.utils.external.utils_pymc import build_baseline_priors, build_sigmoid_term, build_curvature_term,\
                build_final_likelihood
            intercept, sigma = build_baseline_priors(**dim_opt)
            sigmoid_term = build_sigmoid_term(x)
            curvature_term = build_curvature_term(curvature, **dim_opt)

            mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
            likelihood = build_final_likelihood(mu, sigma, y)

            # Sample
            trace = pm.sample_posterior_predictive(trace, var_names=['y', 'sigmoid_term',
                                                                     'curvature_term', 'mu'])

    # Plot
    fig = plot_model_elements(trace)

    # Save
    fig.write_html(out_fname)

    return fig
