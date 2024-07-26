import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import cloudpickle
from matplotlib import pyplot as plt
from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir


def fit_multiple_models(Xy, neuron_name, dataset_name='2022-11-23_worm8',
                        sample_posterior=True, use_additional_behaviors=True, DEBUG=False) -> Tuple[pd.DataFrame, Dict, Dict]:
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
    curvature_terms_to_use = ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']
    if use_additional_behaviors:
        curvature_terms_to_use.extend([#'dorsal_only_body_curvature', 'dorsal_only_head_curvature',
                                      #'ventral_only_body_curvature', 'ventral_only_head_curvature',
                                       'speed', 'self_collision'])
    # First pack into a single dataframe to drop nan, then unpack
    try:
        df_model = get_dataframe_for_single_neuron(Xy, neuron_name, dataset_name=dataset_name,
                                                   curvature_terms=curvature_terms_to_use)
    except KeyError:
        print(f"Skipping {neuron_name} because there is no valid data")
        return None, None, None
    global_manifold = df_model['x'].values
    pca_modes = df_model[['x_pca0', 'x_pca1']].values
    y = df_model['y'].values
    curvature = df_model[curvature_terms_to_use].values

    if df_model.shape[0] == 0:
        print(f"Skipping {neuron_name} because there is no valid data")
        return None, None, None

    if dataset_name == 'all':
        dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
        coords = {'dataset_name': dataset_name_values}
        dims = 'dataset_name'
    else:
        coords = {}
        dims, dataset_name_idx = None, None

    dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)

    with pm.Model(coords=coords) as null_model:
        # Just do a flat line (intercept)
        intercept, sigma = build_baseline_priors()#**dim_opt)
        mu = pm.Deterministic('mu', intercept)
        likelihood = build_final_likelihood(mu, sigma, y)

    with pm.Model(coords=coords) as nonhierarchical_model:
        # Curvature, but no sigmoid
        intercept, sigma = build_baseline_priors()#**dim_opt)
        curvature_term = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, **dim_opt)

        mu = pm.Deterministic('mu', intercept + curvature_term)
        likelihood = build_final_likelihood(mu, sigma, y)

    # with pm.Model(coords=coords) as hierarchical_model:
    #     # Curvature multiplied by sigmoid
    #     intercept, sigma = build_baseline_priors()#**dim_opt)
    #     sigmoid_term = build_sigmoid_term(global_manifold)
    #     curvature_term = build_curvature_term(curvature, **dim_opt)
    #
    #     mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
    #     likelihood = build_final_likelihood(mu, sigma, y)

    with pm.Model(coords=coords) as hierarchical_pca_model:
        # Curvature multiplied by sigmoid
        intercept, sigma = build_baseline_priors()#**dim_opt)
        sigmoid_term = build_sigmoid_term_pca(pca_modes, **dim_opt)
        curvature_term = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, **dim_opt)

        mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
        likelihood = build_final_likelihood(mu, sigma, y)

    coords.update({'time': np.arange(len(y))})
    with pm.Model(coords=coords) as hierarchical_pca_model_with_drift:
        # Curvature multiplied by sigmoid
        intercept, sigma = build_baseline_priors()#**dim_opt)
        sigmoid_term = build_sigmoid_term_pca(pca_modes, **dim_opt)
        curvature_term = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, **dim_opt)
        # num_points = len(y)

        # sigma_alpha = pm.Exponential("sigma_alpha", len(y)/2.0)

        drift_term = pm.GaussianRandomWalk(
            "alpha", sigma=0.01, init_dist=pm.Normal.dist(0, 1), dims="time"
        )
        # drift_term = build_drift_term(**dim_opt)

        # Works if the drift term is gp.latent (VERY SLOW) or a random walk
        mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term + drift_term)
        likelihood = build_final_likelihood(mu, sigma, y)
    #
    #     # Works if the gp is .marginal
    #     eta = 2.0
    #     lengthscale = 500
    #     cov = eta ** 2 * pm.gp.cov.ExpQuad(1, lengthscale)
    #     mu_func = lambda X: intercept + sigmoid_term[X] * curvature_term[X]
    #     # Add white noise to stabilise
    #     cov += pm.gp.cov.WhiteNoise(1e-6)
    #     # Actual gp, then make it a function
    #     gp = pm.gp.Marginal(cov_func=cov, mean_func=mu_func)
    #
    #     likelihood = gp.marginal_likelihood('y', X=np.arange(num_points), y=y, noise=sigma)

    # New: just have rectification with given fwd/rev, not using a sigmoid term
    # fwd_idx, fwd_values = df_model.fwd.factorize()
    # coords = {'fwd': fwd_values}
    # dims = 'fwd'
    # dim_opt = dict(dims=dims, dataset_name_idx=fwd_idx)
    # with pm.Model(coords=coords) as rectified_model:
    #     # Full model
    #     intercept, sigma = build_baseline_priors()
    #     curvature_term = build_curvature_term(curvature, **dim_opt)
    #
    #     mu = pm.Deterministic('mu', intercept + curvature_term)
    #     likelihood = build_final_likelihood(mu, sigma, y)

    # Run inference on final set of models
    # all_models = {'null': null_model,
    #               'nonhierarchical': nonhierarchical_model,
    #               'hierarchical': hierarchical_model,
    #               'rectified': rectified_model,
    #               'hierarchical_pca': hierarchical_pca_model}
    all_models = {'hierarchical_pca': hierarchical_pca_model,
                  'null': null_model,
                  'nonhierarchical': nonhierarchical_model}
    # all_models = {'hierarchical_pca_model_with_drift': hierarchical_pca_model_with_drift,
    #               'hierarchical_pca': hierarchical_pca_model,
    #               'null': null_model,
    #               'nonhierarchical': nonhierarchical_model}
    all_traces = {}
    # base_names_to_sample = {'y', 'sigmoid_term', 'curvature_term', 'phase_shift', 'sigmoid_slope'}
    for name, model in all_models.items():
        with model:
            opt = dict(draws=1000, tune=1000, random_seed=rng, target_accept=0.96)
            if DEBUG:
                opt['draws'] = 10
                opt['tune'] = 10

            trace = pm.sample(**opt,
                              chains=4, return_inferencedata=True, idata_kwargs={"log_likelihood": True})
            if sample_posterior:
                posterior_keys = list(trace.posterior.keys())
                # var_names = base_names_to_sample.intersection(posterior_keys)
                # Keep only those that have 'term' in it, because those are the time series
                posterior_keys = [key for key in posterior_keys if 'term' in key]
                posterior_keys.extend(['y', 'mu'])
                print(f"Sampling posterior predictive for {name}: {posterior_keys}")
                trace.extend(pm.sample_posterior_predictive(trace, random_seed=rng, progressbar=False,
                                                            var_names=posterior_keys))

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
        sigma = pm.HalfCauchy("sigma", beta=0.02)

    else:
        # Include hyperprior
        hyper_intercept = pm.Normal('hyper_intercept', mu=0, sigma=1)
        hyper_intercept_sigma = pm.Exponential('hyper_intercept_sigma', lam=1)
        zscore_intercept = pm.Normal('zscore_intercept', mu=0, sigma=1, dims=dims)
        intercept = pm.Deterministic('intercept', hyper_intercept + zscore_intercept*hyper_intercept_sigma)[dataset_name_idx]

        # Also vary sigma per dataset; simpler because we don't have to zscore it
        sigma = pm.HalfCauchy("sigma", beta=0.02, dims=dims)[dataset_name_idx]

    return intercept, sigma


def build_final_likelihood(mu, sigma, y, nu=100):
    return pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)


def build_sigmoid_term(x, force_positive_slope=True):
    # Sigmoid (hierarchy) term
    if force_positive_slope:
        log_sigmoid_slope = pm.Normal('log_sigmoid_slope', mu=0, sigma=1)  # Using log-amplitude for positivity
        sigmoid_slope = pm.Deterministic('sigmoid_slope', pm.math.exp(log_sigmoid_slope))
    else:
        sigmoid_slope = pm.Normal('sigmoid_slope', mu=0, sigma=1)
    inflection_point = pm.Normal('inflection_point', mu=0, sigma=2)
    # Sigmoid term
    sigmoid_term = pm.Deterministic('sigmoid_term', pm.math.sigmoid(sigmoid_slope * (x - inflection_point)))
    return sigmoid_term


def build_sigmoid_term_pca(x_pca_modes, force_positive_slope=True, dims=None, dataset_name_idx=None):
    # Sigmoid (hierarchy) term
    # if force_positive_slope:
    #     log_sigmoid_slope = pm.Normal('log_sigmoid_slope', mu=0, sigma=1)  # Using log-amplitude for positivity
    #     sigmoid_slope = pm.Deterministic('sigmoid_slope', pm.math.exp(log_sigmoid_slope))
    # else:
    #     sigmoid_slope = pm.Normal('sigmoid_slope', mu=0, sigma=1)
    inflection_point = pm.Normal('inflection_point', mu=0, sigma=2)

    # PCA modes and coefficients
    if dims is None:
        hyper_pca_amplitude, hyper_pca_sigma = np.array([0, 0]), np.array([1, 1])
        zscore_pca_amplitude = pm.Normal('zscore_pca_amplitude', mu=0, sigma=1, dims=dims)
        pca_amplitude = pm.Deterministic('pca_amplitude',
                                         hyper_pca_amplitude + zscore_pca_amplitude*hyper_pca_sigma)
        pca_term = pm.Deterministic('pca_term', pm.math.dot(x_pca_modes, pca_amplitude))
    else:
        # Hyperprior
        hyper_pca0_amplitude = pm.Normal('hyper_pca0_amplitude', mu=0, sigma=1)
        hyper_pca0_sigma = pm.Exponential('hyper_pca0_sigma', lam=1)
        hyper_pca1_amplitude = pm.Normal('hyper_pca1_amplitude', mu=0, sigma=1)
        hyper_pca1_sigma = pm.Exponential('hyper_pca1_sigma', lam=1)
        zscore_pca0_amplitude = pm.Normal('zscore_pca0_amplitude', mu=0, sigma=1, dims=dims)
        zscore_pca1_amplitude = pm.Normal('zscore_pca1_amplitude', mu=0, sigma=1, dims=dims)

        pca0_amplitude = pm.Deterministic('pca0_amplitude',
                                          hyper_pca0_amplitude + zscore_pca0_amplitude*hyper_pca0_sigma)
        pca1_amplitude = pm.Deterministic('pca1_amplitude',
                                          hyper_pca1_amplitude + zscore_pca1_amplitude*hyper_pca1_sigma)
        # Multiply them separately
        pca_term = pm.Deterministic('pca_term',
                                    pca0_amplitude[dataset_name_idx] * x_pca_modes[:, 0] +
                                    pca1_amplitude[dataset_name_idx] * x_pca_modes[:, 1])

    # Put it together Sigmoid term
    sigmoid_term = pm.Deterministic('sigmoid_term', pm.math.sigmoid(pca_term - inflection_point))
    return sigmoid_term


def build_curvature_term(curvature, curvature_terms_to_use=None, dims=None, dataset_name_idx=None):
    if curvature_terms_to_use is None:
        assert curvature.shape[1] == 4, f"Default curvature terms are for 4 eigenworms, found {curvature.shape[1]}"
        curvature_terms_to_use = ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']
    # Alternative: sample directly from the phase shift and amplitude, then convert into coefficients
    # This assumes that eigenworms 1 and 2 are approximately a sine and cosine wave, and puts it into polar coordinates
    phase_shift = pm.Uniform('phase_shift', lower=-np.pi, upper=np.pi, transform=pm.distributions.transforms.circular)
    if dims is None:
        hyper_log_amplitude, hyper_log_sigma = 0, 1
    else:
        # Hyperprior
        hyper_log_amplitude = pm.Normal('log_amplitude_mu', mu=0, sigma=1)
        hyper_log_sigma = pm.Exponential('log_amplitude_sigma', lam=1)
    zscore_log_amplitude = pm.Normal('zscore_log_amplitude', mu=0, sigma=1, dims=dims)
    log_amplitude = pm.Deterministic('log_amplitude', hyper_log_amplitude + zscore_log_amplitude*hyper_log_sigma)
    amplitude = pm.Deterministic('amplitude', pm.math.exp(log_amplitude))
    # There is a positive and negative solution, so choose the positive one for the first term
    eigenworm1_coefficient = pm.Deterministic('eigenworm1_coefficient', amplitude * pm.math.cos(phase_shift))
    eigenworm2_coefficient = pm.Deterministic('eigenworm2_coefficient', -amplitude * pm.math.sin(phase_shift))
    # The rest are not part of the sine/cosine pair, but we aren't sure how many there are
    additional_column_dict = {}
    if len(curvature_terms_to_use) > 2:
        for col_name in curvature_terms_to_use[2:]:
            # None of these terms are modulated per dataset
            # If the column name is like "eigenworm3", the coefficient name is "eigenworm4_coefficient"
            # Because we want to start at 1, not 0
            if col_name.startswith('eigenworm'):
                coef_name = f'eigenworm{int(col_name[-1])+1}_coefficient'
            else:
                coef_name = f'{col_name}_coefficient'
            additional_column_dict[coef_name] = pm.Normal(coef_name, mu=0, sigma=0.5, dims=None)
    # eigenworm3_coefficient = pm.Normal('eigenworm3_coefficient', mu=0, sigma=0.5, dims=None)
    # eigenworm4_coefficient = pm.Normal('eigenworm4_coefficient', mu=0, sigma=0.5, dims=None)

    if dims is None:
        all_cols = [eigenworm1_coefficient, eigenworm2_coefficient]#, eigenworm3_coefficient, eigenworm4_coefficient]
        all_cols.extend(list(additional_column_dict.values()))  # Don't need to worry about the order
        coefficients_vec = pm.Deterministic('coefficients_vec', pm.math.stack(all_cols))
        curvature_term = pm.Deterministic('curvature_term', pm.math.dot(curvature, coefficients_vec))
    else:
        # Multiply them separately, but do not subindex by dataset for other terms
        curvature_term = pm.Deterministic('curvature_term',
                                          eigenworm1_coefficient[dataset_name_idx] * curvature[:, 0] +
                                          eigenworm2_coefficient[dataset_name_idx] * curvature[:, 1] +
                                          np.sum([coef * curvature[:, i+2] for i, coef
                                                  in enumerate(additional_column_dict.values())])
                                          )
    return curvature_term


def build_drift_term_gp(n, lengthscale=None, dims=None, dataset_name_idx=None):
    # Drift term (gaussian process)
    if lengthscale is None:
        lengthscale = n / 3.0
    eta = 2.0
    cov = eta ** 2 * pm.gp.cov.ExpQuad(1, lengthscale)
    # Add white noise to stabilise
    cov += pm.gp.cov.WhiteNoise(1e-6)
    # Actual gp, then make it a function
    gp = pm.gp.Latent(cov_func=cov)  # VERY slow
    X = np.linspace(0, n, n)[:, None]  # The inputs to the GP must be arranged as a column vector
    drift_term = gp.prior("f", X=X)

    return drift_term


def build_drift_term(dims=None, dataset_name_idx=None):
    # Drift term (random walk); needs 'time' in the dims
    # std of random walk
    sigma_alpha = pm.Exponential("sigma_alpha", 1.0)

    drift_term = pm.GaussianRandomWalk(
        "alpha", sigma=sigma_alpha, init_dist=pm.Normal.dist(0, 1), dims="time"
    )

    return drift_term


def get_dataframe_for_single_neuron(Xy, neuron_name, curvature_terms=None,
                                    dataset_name='all', additional_columns=None):
    if dataset_name != 'all':
        _Xy = Xy[Xy['dataset_name'] == dataset_name]
    else:
        _Xy = Xy
    if curvature_terms is None:
        curvature_terms = ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']
    # First, extract data, z-score, and drop na values
    # Allow gating based on the global component
    x = _Xy[f'{neuron_name}_manifold']
    x = (x - x.mean()) / x.std()  # z-score
    # Alternative: include the pca modes
    x_pca0 = _Xy[f'pca_0']
    x_pca0 = (x_pca0 - x_pca0.mean()) / x_pca0.std()  # z-score
    x_pca1 = _Xy[f'pca_1']
    x_pca1 = (x_pca1 - x_pca1.mean()) / x_pca1.std()  # z-score
    # Just predict the residual
    y = _Xy[f'{neuron_name}'] - _Xy[f'{neuron_name}_manifold']
    y = (y - y.mean()) / y.std()  # z-score
    # Interesting covariate
    curvature = _Xy[curvature_terms]
    curvature = (curvature - curvature.mean()) / curvature.std()  # z-score
    # State
    fwd = _Xy['fwd'].astype(str)
    # Package as dataframe again, and drop na values
    all_dfs = [pd.DataFrame({'y': y, 'x': x, 'x_pca0': x_pca0, 'x_pca1': x_pca1,
                             'dataset_name': _Xy['dataset_name'], 'fwd': fwd}),
               pd.DataFrame(curvature)]
    if additional_columns is not None:
        all_dfs.append(_Xy[additional_columns])
    df_model = pd.concat(all_dfs, axis=1)
    df_model = df_model.dropna()
    return df_model


def main(neuron_name=None, do_gfp=False, dataset_name='all', skip_if_exists=True):
    """
    Runs for hardcoded data location for a single neuron

    Saves all the information in the same directory as the data, in the 'output' subdirectory

    Commonly used with:
        dataset_name = 'all' to run the neuron for all datasets at once
        dataset_name = 'loop' to run that neuron for each dataset seperately

    Returns
    -------

    """
    if neuron_name is None:
        neuron_name = 'VB02'
    print(f"Running all 3 bayesian models for {neuron_name} with do_gfp={do_gfp}")

    data_dir = get_hierarchical_modeling_dir(do_gfp)
    fname = os.path.join(data_dir, 'data.h5')
    Xy = pd.read_hdf(fname)
    print(f"Loaded data from {fname}")

    if dataset_name == 'loop':
        # Loop over all datasets
        for dataset_name in Xy['dataset_name'].unique():
            if dataset_name == 'loop':
                # Recursion error
                continue
            main(neuron_name, do_gfp=do_gfp, dataset_name=dataset_name, skip_if_exists=skip_if_exists)
        return

    if dataset_name == 'all':
        output_dir = os.path.join(data_dir, 'output')
    else:
        output_dir = os.path.join(data_dir, 'output_single_dataset')
    Path(output_dir).mkdir(exist_ok=True)
    # Check if it already exists
    if skip_if_exists and os.path.exists(os.path.join(output_dir, f'{neuron_name}_loo.h5')):
        print(f"Skipping {neuron_name} because it already exists")
        return

    # Fit models
    df_compare, all_traces, all_models = fit_multiple_models(Xy, neuron_name, dataset_name=dataset_name)

    if df_compare is None:
        print(f"Skipping {neuron_name} because there is no valid data")
        return

    save_all_model_outputs(dataset_name, neuron_name, df_compare, all_traces, all_models, output_dir)


def save_all_model_outputs(dataset_name, neuron_name, df_compare, all_traces, all_models, output_dir):
    # Save objects
    if dataset_name == 'all':
        output_fname_base = f'{neuron_name}'

        # arviz has a specific function for traces
        for model_name, traces in all_traces.items():
            az.to_netcdf(traces, os.path.join(output_dir, f'{output_fname_base}_{model_name}_trace.nc'))
    else:
        output_fname_base = f'{neuron_name}_{dataset_name}'
    # Also save the model
    # See https://discourse.pymc.io/t/how-save-pymc-v5-models/13022
    model_fname = os.path.join(output_dir, f'{output_fname_base}_model.cloud_pkl')
    with open(model_fname, 'wb') as buffer:
        cloudpickle.dump(all_models, buffer)
    # Only save for the all dataset version
    fname = os.path.join(output_dir, f'{output_fname_base}_loo.h5')
    df_compare.to_hdf(fname, key='df_with_missing')
    # Save plots
    az.plot_compare(df_compare, insample_dev=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_fname_base}_model_comparison.png'))
    plt.close()
    print(f"Saved all objects for {neuron_name} in {output_dir}")


if __name__ == '__main__':
    # Get neuron name from argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_name', '-n', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--do_gfp', type=str)
    args = parser.parse_args()

    # Parse do_gfp into a boolean
    if args.do_gfp is None:
        do_gfp = False
    else:
        do_gfp = args.do_gfp.lower() == 'true'

    main(neuron_name=args.neuron_name, do_gfp=do_gfp)
