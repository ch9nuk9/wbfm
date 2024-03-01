import os
import pickle
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt
from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir


def fit_multiple_models(Xy, neuron_name, dataset_name='2022-11-23_worm8') -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Fit multiple models to the same data, to be used for model comparison

    Parameters
    ----------
    Xy
    neuron_name

    Returns
    -------

    """
    rng = 424242

    if dataset_name == 'all':
        # First pack into a single dataframe for preprocessing, then unpack
        df_model = get_dataframe_for_single_neuron(Xy, neuron_name)
        x = df_model['x'].values
        y = df_model['y'].values
        curvature = df_model[['eigenworm0', 'eigenworm1', 'eigenworm2']].values

        dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
        coords = {'dataset_name': dataset_name_values}
        dims = 'dataset_name'
    else:
        # Unpack data into x, y, and curvature
        # For now, just use one dataset
        ind_data = Xy['dataset_name'] == dataset_name
        # Allow gating based on the global component
        x = Xy[f'{neuron_name}_manifold'][ind_data].values
        x = (x - x.mean()) / x.std()  # z-score

        if pd.Series(x).count() == 0:
            print(f"Skipping {neuron_name} because there is no valid data")
            return None, None, None

        # Just predict the residual
        y = Xy[f'{neuron_name}'][ind_data].values - Xy[f'{neuron_name}_manifold'][ind_data].values
        y = (y - y.mean()) / y.std()  # z-score

        # Interesting covariate
        curvature = Xy[['eigenworm0', 'eigenworm1', 'eigenworm2']][ind_data].values
        curvature = (curvature - curvature.mean()) / curvature.std()  # z-score

        coords = {}
        dims, dataset_name_idx = None, None

    dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)

    with pm.Model(coords=coords) as null_model:
        # Just do a flat line (intercept)
        intercept, sigma = build_baseline_priors(**dim_opt)
        mu = pm.Deterministic('mu', intercept)
        likelihood = build_final_likelihood(mu, sigma, y)

    with pm.Model(coords=coords) as nonhierarchical_model:
        # Everything except sigmoid
        intercept, sigma = build_baseline_priors(**dim_opt)
        curvature_term = build_curvature_term(curvature, **dim_opt)

        mu = pm.Deterministic('mu', intercept + curvature_term)
        likelihood = build_final_likelihood(mu, sigma, y)

    with pm.Model(coords=coords) as hierarchical_model:
        # Full model
        intercept, sigma = build_baseline_priors(**dim_opt)
        sigmoid_term = build_sigmoid_term(x)
        curvature_term = build_curvature_term(curvature, **dim_opt)

        mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
        likelihood = build_final_likelihood(mu, sigma, y)

    # Run inference on all models
    all_models = {'hierarchical': hierarchical_model, 'nonhierarchical': nonhierarchical_model, 'null': null_model}
    all_traces = {}
    for name, model in all_models.items():
        with model:
            trace = pm.sample(1000, tune=1000, cores=4, return_inferencedata=True, #target_accept=0.95,
                              idata_kwargs={"log_likelihood": True}, random_seed=rng)
            all_traces[name] = trace

    # Compute model comparisons
    all_loo = {}
    for name, trace in all_traces.items():
        loo = az.loo(trace)
        all_loo[name] = loo

    df_compare = az.compare(all_loo)

    return df_compare, all_traces, all_models


def build_baseline_priors(dims=None, dataset_name_idx=None):
    if dims is None:
        intercept = pm.Normal('intercept', mu=0, sigma=1)
    else:
        # Include hyperprior
        hyper_intercept = pm.Normal('hyper_intercept', mu=0, sigma=1)
        hyper_intercept_sigma = pm.Exponential('hyper_intercept_sigma', lam=1)
        zscore_intercept = pm.Normal('z_intercept', mu=0, sigma=1, dims=dims)
        intercept = pm.Deterministic('intercept', hyper_intercept + zscore_intercept*hyper_intercept_sigma)[dataset_name_idx]
        # intercept = pm.Normal('intercept', mu=hyper_intercept, sigma=hyper_intercept_sigma, dims=dims)[dataset_name_idx]
    sigma = pm.HalfCauchy("sigma", beta=0.02)
    return intercept, sigma


def build_final_likelihood(mu, sigma, y, nu=100):
    return pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)


def build_sigmoid_term(x):
    # Sigmoid (hierarchy) term
    log_sigmoid_slope = pm.Normal('log_sigmoid_slope', mu=0, sigma=1)  # Using log-amplitude for positivity
    sigmoid_slope = pm.Deterministic('sigmoid_slope', pm.math.exp(log_sigmoid_slope))
    inflection_point = pm.Normal('inflection_point', mu=0, sigma=2)
    # Sigmoid term
    sigmoid_term = pm.Deterministic('sigmoid_term', pm.math.sigmoid(sigmoid_slope * (x - inflection_point)))
    return sigmoid_term


def build_curvature_term(curvature, dims=None, dataset_name_idx=None):
    # Alternative: sample directly from the phase shift and amplitude, then convert into coefficients
    # This assumes that eigenworms 1 and 2 are approximately a sine and cosine wave
    # See trig identities: https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Linear_combinations
    # And this for solving the equations: https://www.wolframalpha.com/input?i=Solve+c%3Dsign%28a%29sqrt%28a%5E2%2Bb%5E2%29+and+phi%3Darctan%28-b%2Fa%29+for+a+and+b
    phase_shift = pm.Uniform('phase_shift', lower=-np.pi, upper=np.pi, transform=pm.distributions.transforms.circular)
    if dims is None:
        hyper_log_amplitude, hyper_log_sigma = 0, 1
    else:
        # Hyperprior
        hyper_log_amplitude = pm.Normal('log_amplitude_mu', mu=0, sigma=1)
        hyper_log_sigma = pm.Exponential('log_amplitude_sigma', lam=1)
    zscore_log_amplitude = pm.Normal('z_log_amplitude', mu=0, sigma=1, dims=dims)
    log_amplitude = pm.Deterministic('log_amplitude', hyper_log_amplitude + zscore_log_amplitude*hyper_log_sigma)
    amplitude = pm.Deterministic('amplitude', pm.math.exp(log_amplitude))
    # There is a positive and negative solution, so choose the positive one for the first term
    eigenworm1_coefficient = pm.Deterministic('eigenworm1_coefficient', amplitude * pm.math.cos(phase_shift))
    eigenworm2_coefficient = pm.Deterministic('eigenworm2_coefficient', -amplitude * pm.math.sin(phase_shift))
    # This one is not part of the sine/cosine pair
    eigenworm3_coefficient = pm.Normal('eigenworm3_coefficient', mu=0, sigma=0.5, dims=None)

    if dims is None:
        coefficients_vec = pm.Deterministic('coefficients_vec', pm.math.stack([eigenworm1_coefficient,
                                                                               eigenworm2_coefficient,
                                                                               eigenworm3_coefficient]))
        curvature_term = pm.Deterministic('curvature_term', pm.math.dot(curvature, coefficients_vec))
    else:

        # Multiply them separately
        curvature_term = pm.Deterministic('curvature_term',
                                          eigenworm1_coefficient[dataset_name_idx] * curvature[:, 0] +
                                          eigenworm2_coefficient[dataset_name_idx] * curvature[:, 1] +
                                          eigenworm3_coefficient * curvature[:, 2])
    return curvature_term


def build_multidataset_model(Xy, neuron_name):
    """
    Builds the main model, but with certain variables stratefied by dataset

    Parameters
    ----------
    Xy
    neuron_name

    Returns
    -------

    """

    df_model = get_dataframe_for_single_neuron(Xy, neuron_name)
    dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
    coords = {'dataset_name': dataset_name_values}

    # Build model
    with pm.Model(coords=coords) as model:
        # Group-level random effect: intercept
        hyper_intercept = pm.Normal('hyper_intercept', mu=0, sigma=1)
        zscore_intercept = pm.Normal('z_intercept', mu=0, sigma=1, dims='dataset_name')
        intercept = pm.Deterministic('intercept', hyper_intercept + zscore_intercept)

        # First try: pooling for sigmoid term
        x = df_model['x'].values
        sigmoid_term = build_sigmoid_term(x)

        # Also pool curvature
        curvature = df_model[['eigenworm0', 'eigenworm1', 'eigenworm2']].values
        curvature_term = build_curvature_term(curvature, dims='dataset_name', dataset_name_idx=dataset_name_idx)

        # Expected value of outcome
        mu = pm.Deterministic('mu', intercept[dataset_name_idx] + sigmoid_term * curvature_term)

        # Likelihood
        sigma = pm.HalfCauchy("sigma", beta=0.02)

        y = df_model['y'].values
        likelihood = build_final_likelihood(mu, sigma, y)

    return model, df_model


def get_dataframe_for_single_neuron(Xy, neuron_name):
    # First, extract data, z-score, and drop na values
    # Allow gating based on the global component
    x = Xy[f'{neuron_name}_manifold']
    x = (x - x.mean()) / x.std()  # z-score
    # Just predict the residual
    y = Xy[f'{neuron_name}'] - Xy[f'{neuron_name}_manifold']
    y = (y - y.mean()) / y.std()  # z-score
    # Interesting covariate
    curvature = Xy[['eigenworm0', 'eigenworm1', 'eigenworm2']]
    curvature = (curvature - curvature.mean()) / curvature.std()  # z-score
    # Package as dataframe again, and drop na values
    df_model = pd.concat([pd.DataFrame({'y': y, 'x': x, 'dataset_name': Xy['dataset_name']}), pd.DataFrame(curvature)],
                         axis=1)
    df_model = df_model.dropna()
    return df_model


def main(neuron_name, skip_if_exists=True):
    """
    Runs for hardcoded data location for a single neuron

    Saves all the information in the same directory as the data, with a subfolder per neuron

    Returns
    -------

    """
    data_dir = get_hierarchical_modeling_dir()
    fname = os.path.join(data_dir, 'data.h5')
    Xy = pd.read_hdf(fname)

    output_dir = os.path.join(data_dir, 'output')
    Path(output_dir).mkdir(exist_ok=True)
    # Check if it already exists
    if skip_if_exists and os.path.exists(os.path.join(output_dir, f'{neuron_name}_loo.h5')):
        print(f"Skipping {neuron_name} because it already exists")
        return

    # Fit models
    df_compare, all_traces, all_models = fit_multiple_models(Xy, neuron_name, dataset_name='all')

    if df_compare is None:
        print(f"Skipping {neuron_name} because there is no valid data")
        return

    # Save objects
    # all_loo is just a dictionary of dfs, so save it as a pickle without looping
    fname = os.path.join(output_dir, f'{neuron_name}_loo.h5')
    df_compare.to_hdf(fname, key='df_with_missing')

    # Save model directly with pickle
    # fname = os.path.join(output_dir, f'{neuron_name}_models.pkl')
    # with open(fname, 'wb') as f:
    #     pickle.dump(all_models, f)

    # arviz has a specific function for traces
    for model_name, traces in all_traces.items():
        az.to_netcdf(traces, os.path.join(output_dir, f'{neuron_name}_{model_name}_trace.nc'))

    # Save plots
    az.plot_compare(df_compare, insample_dev=False)
    plt.savefig(os.path.join(output_dir, f'{neuron_name}_model_comparison.png'))
    plt.close()

    print(f"Saved all objects for {neuron_name} in {output_dir}")


if __name__ == '__main__':
    # Get neuron name from argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('neuron_name', type=str)
    args = parser.parse_args()

    main(args.neuron_name)
