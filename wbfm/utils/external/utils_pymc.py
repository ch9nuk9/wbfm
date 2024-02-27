import numpy as np
import pymc as pm
import arviz as az


def fit_multiple_models(Xy, neuron_name):
    """
    Fit multiple models to the same data, to be used for model comparison

    Parameters
    ----------
    Xy
    neuron_name

    Returns
    -------

    """

    # Unpack data into x, y, and curvature
    # For now, just use one dataset

    ind_data = Xy['dataset_name'] == '2022-11-23_worm8'
    # Allow gating based on the global component
    x = Xy[f'{neuron_name}_manifold'][ind_data].values
    x = (x - x.mean()) / x.std()  # z-score

    # Just predict the residual
    y = Xy[f'{neuron_name}'][ind_data].values - Xy[f'{neuron_name}_manifold'][ind_data].values
    y = (y - y.mean()) / y.std()  # z-score

    # Interesting covariate
    # curvature = Xy['vb02_curvature'][ind_data].values
    curvature = Xy[['eigenworm0', 'eigenworm1', 'eigenworm2']][ind_data].values
    # curvature = Xy[['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']][ind_data].values
    curvature = (curvature - curvature.mean()) / curvature.std()  # z-score

    with pm.Model() as complex_model:
        # Priors for parameters
        # Sigmoid (hierarchy) term
        log_sigmoid_slope = pm.Normal('log_sigmoid_slope', mu=0, sigma=1)  # Using log-amplitude for positivity
        inflection_point = pm.Normal('inflection_point', mu=0, sigma=2)
        log_amplitude = pm.Normal('log_amplitude', mu=0, sigma=2)  # Using log-amplitude for positivity
        # Baseline term
        intercept = pm.Normal('intercept', mu=0, sigma=10)

        # Transforming log-amplitude to ensure positivity
        amplitude = pm.Deterministic('amplitude', pm.math.exp(log_amplitude))
        sigmoid_slope = pm.Deterministic('sigmoid_slope', pm.math.exp(log_sigmoid_slope))

        # Sigmoid term
        sigmoid_term = pm.Deterministic('sigmoid_term', pm.math.sigmoid(sigmoid_slope * (x - inflection_point)))

        # Alternative: Define covariance matrix to enforce sum of squares constraint
        num_coefficients = curvature.shape[1]
        covariance_matrix = np.eye(num_coefficients) / num_coefficients
        coefficients_vec = pm.MvNormal('coefficients_vec', mu=0, cov=covariance_matrix, shape=num_coefficients)

        curvature_term = pm.Deterministic('curvature_term', pm.math.dot(curvature, coefficients_vec))

        # Expected value of outcome
        mu = pm.Deterministic('mu', intercept + amplitude * sigmoid_term * curvature_term)

        # Likelihood
        sigma = pm.HalfNormal('sigma', sigma=1)
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

    with pm.Model() as null_model:
        # Just do a flat line (intercept)
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)
        likelihood = pm.Normal('y', mu=intercept, sigma=sigma, observed=y)

    # Run inference on all models
    all_models = [complex_model, null_model]
    all_traces = []
    for model in all_models:
        with model:
            trace = pm.sample(1000, tune=1000, cores=64, target_accept=0.99, return_inferencedata=True,
                              idata_kwargs={"log_likelihood": True})
            all_traces.append(trace)

    # Compute model comparisons
    all_loo = []
    for i, trace in enumerate(all_traces):
        loo = az.loo(trace)
        all_loo.append(loo)

    df_compare = az.compare({'complex': all_loo[0], 'null': all_loo[1]})

    return df_compare, all_traces, all_models
