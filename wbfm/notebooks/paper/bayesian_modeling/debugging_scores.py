#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import pandas as pd
import plotly.express as px
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import os
import arviz as az
from wbfm.utils.general.hardcoded_paths import get_hierarchical_modeling_dir
# fname = 'data.h5'

fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
Xy = pd.read_hdf(fname)

fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
Xy_gfp = pd.read_hdf(fname)


from wbfm.utils.external.utils_arviz import plot_model_elements
from wbfm.utils.external.utils_pymc import get_dataframe_for_single_neuron, build_baseline_priors, build_sigmoid_term, build_curvature_term, build_final_likelihood, build_sigmoid_term_pca


# # GFP: shouldn't have an improvement from hierarchical model

# folder_name = '/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling_gfp/output/'
folder_name = '/lisc/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/output/'
base_fname = os.path.join(folder_name, 'VB02_{}_trace.nc')
suffixes = ['null', 'hierarchical_pca', 'nonhierarchical']
all_traces = {}
for s in suffixes:
    fname = base_fname.format(s)
    all_traces[s] = az.from_netcdf(fname)

fname = base_fname.replace('{}_trace.nc', 'loo.h5')
df_compare = pd.read_hdf(fname)

# import cloudpickle
# fname = base_fname.replace('{}_trace.nc', 'model.cloud_pkl')
# with open(fname, 'rb') as buffer:
#      all_models = cloudpickle.load(buffer)


df_compare


scaling = len(all_traces['hierarchical_pca'].observed_data.y)
# az.plot_compare(np.divide(df_compare, scaling, where=df_compare.dtypes.ne(object)).combine_first(df_compare))
cols = ['warning', 'scale', 'rank']
df_compare2 = df_compare.copy().drop(columns=cols)
df_compare2 = df_compare2 / scaling
df_compare2[cols] = df_compare[cols]
az.plot_compare(df_compare2)


len(all_traces['hierarchical_pca'].observed_data.y)


# az.plot_loo_pit(idata=all_traces['hierarchical_pca'], y="y", ecdf=True)


# az.plot_bf(all_traces['hierarchical_pca'], var_name='inflection_point')


# Takes a long time and isn't zoomed in; shows every data point
# az.plot_elpd(all_traces)


az.plot_loo_pit(all_traces['hierarchical_pca'], y='y')


# Isn't zoomed in; shows every data point
loo = az.loo(all_traces['hierarchical_pca'], pointwise=True)
az.plot_khat(loo)


# fig_raw = plot_model_elements(all_traces['hierarchical_pca'], to_show=True,
#                          y_list=['sigmoid_term', 'curvature_term'])

# # fig_raw.update_xaxes(range=Xy_ind_range.iloc[12, :])


from wbfm.utils.external.utils_arviz import plot_ts
fig = plot_ts(all_traces['hierarchical_pca'])


# Plot posterior distributions
var_names = ['intercept', 'inflection_point', #'amplitude', 
             #'phase_shift', 
             'eigenworm3', 'eigenworm4',
             'self', 'speed']
az.plot_forest(all_traces['hierarchical_pca'], var_names=var_names, filter_vars='like',
              kind='ridgeplot', 
               combined=True);


# Plot posterior distributions
var_names = [#'intercept', 'inflection_point', #'amplitude', 
             'phase_shift']
az.plot_forest(all_traces['hierarchical_pca'], var_names=var_names, filter_vars='like',
              kind='ridgeplot', combined=True);



# var_names = ['intercept', 'inflection_point', 'amplitude', 
#              'phase_shift', 'eigenworm3_coefficient']
# az.plot_trace(all_traces['hierarchical_pca'], compact=True, var_names=var_names);


# # Debug: look at gaussian process prior

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

lengthscale = 500
eta = 2.0
cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)
# Add white noise to stabilise
cov += pm.gp.cov.WhiteNoise(1e-6)

n = 1600
X = np.linspace(0, n, n)[:, None]
K = cov(X).eval()

plt.plot(
    X,
    pm.draw(
        pm.MvNormal.dist(mu=np.zeros(len(K)), cov=K, shape=K.shape[0]), draws=3, random_seed=rng
    ).T,
)
plt.title("Samples from the GP prior")
plt.ylabel("y")
plt.xlabel("X");


# import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt

# # Sample data
# n = 100
# X = np.linspace(0, 10, n)
# true_slope = 0.5
# regressor = X  # Example regressor
# true_mean = np.sin(X) + true_slope * regressor
# y = np.random.normal(true_mean, 0.1)

# # Model
# with pm.Model() as model:
#     # Define the Gaussian Process
#     length_scale = 1.0
#     eta = 1.0
    
#     # Covariance function
#     cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=length_scale)
    
#     # Latent variable for the slope
#     slope = pm.Normal("slope", mu=0, sigma=1)
    
#     # Define the mean function which includes the regressor and latent slope
#     def mean_function(X, slope):
#         return slope * X.flatten()  # Adjust this as per your regression model

#     # GP with the mean function that includes the regressor and latent slope
#     gp = pm.gp.Marginal(mean_func=lambda X: mean_function(X, slope), cov_func=cov_func)
    
#     # Likelihood: Normal distribution whose mean is given by the GP
#     sigma = 0.1
#     y_obs = gp.marginal_likelihood("y_obs", X=X[:, None], y=y, noise=sigma)

#     # Inference
#     trace = pm.sample(1000, return_inferencedata=True)

# # Plot the results
# with model:
#     gp_samples = pm.sample_posterior_predictive(trace, var_names=["y_obs"])


# Visualize
plt.figure(figsize=(10, 6))
plt.plot(X, y, "ok", ms=3, alpha=0.5, label="Data")
plt.plot(X, true_mean, "r", lw=1.5, label="True mean")
plt.plot(X, gp_samples.posterior_predictive["y_obs"].mean(axis=1).T, "b", alpha=0.1, label='gp_samples')
plt.xlabel("X")
plt.ylabel("y")
plt.title("GP Posterior Samples for the Mean Function with Regressor and Latent Slope")
plt.legend()
plt.show()


gp_samples.posterior_predictive["y_obs"].mean(axis=1).shape

