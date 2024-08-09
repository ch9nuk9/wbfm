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


# [n for n in Xy.columns.sort_values() if ('neuron' not in n and 'manifold' not in n)]


# [n for n in Xy_gfp.columns.sort_values() if ('neuron' not in n and 'manifold' not in n)]


Xy = Xy.rename(columns={0: 'pca_0', 1: 'pca_1'})
Xy.to_hdf(fname, key='df_with_missing')


# Xy


# ## Check that the eignworms seem phase aligned

px.line(Xy.drop(columns='dataset_name')/Xy.drop(columns='dataset_name').std(), y=['VB02', 'eigenworm1', 'vb02_curvature'])





import pymc as pm

ind_data = Xy['dataset_name'] == '2022-11-23_worm8'

# Allow gating based on the global component
x = Xy['VB02_manifold'][ind_data].values
x = (x-x.mean()) / x.std()  # z-score

# Just predict the residual
y = Xy['VB02'][ind_data].values - Xy['VB02_manifold'][ind_data].values
y = (y-y.mean()) / y.std()  # z-score

# Interesting covariate
curvature = Xy['vb02_curvature'][ind_data].values
curvature = (curvature-curvature.mean()) / curvature.std()  # z-score

print(x.shape, y.shape, curvature.shape)


with pm.Model() as model:
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
    # sigmoid_term = pm.Deterministic('sigmoid_term', sigmoid(x, inflection_point))

    # Expected value of outcome
    mu = intercept + amplitude * sigmoid_term * curvature

    # Likelihood
    sigma = pm.HalfNormal('sigma', sigma=1)
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)


pm.model_to_graphviz(model)


# Run inference
with model:
    trace = pm.sample(1000, tune=1000, cores=64)


# Plot posterior distributions
az.plot_trace(trace, compact=True, var_names=['sigmoid_slope', 'inflection_point', 'amplitude']);


import matplotlib.pyplot as plt
# Generate posterior predictive samples
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['y', 'sigmoid_term'])


# Average across chains, but not draws
y_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['y'], axis=0), axis=0)
sigmoid_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['sigmoid_term'], axis=0), axis=0)


df_pred = pd.DataFrame({'y': y, 'y_pred': y_pred, 'curvature': curvature, 'manifold': x,
                        'fwd': Xy['fwd'][ind_data], 'sigmoid_pred': sigmoid_pred})
px.line(df_pred)


# # Look at the other measures of behavior

curvature_terms_to_use = ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3',
                              'dorsal_only_body_curvature', 'dorsal_only_head_curvature',
                              'ventral_only_body_curvature', 'ventral_only_head_curvature', 'self_collision', 'speed']
df_beh = Xy[curvature_terms_to_use].copy()
df_beh = (df_beh-df_beh.mean())/df_beh.std()

df_corr = df_beh.corr()
px.imshow(df_corr - np.eye(len(df_corr)))


from sklearn.decomposition import PCA
from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
pca = PCA(n_components=8).fit(fill_nan_in_dataframe(df_beh))
px.scatter(pca.explained_variance_ratio_)








# # DEBUG: Build model in jupyter

from wbfm.utils.external.utils_pymc import build_baseline_priors, build_final_likelihood, build_sigmoid_term, build_curvature_term


Xy['VB02_manifold'].count(), Xy['VB02'].count(), Xy[['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']].count()



use_additional_behaviors = True
curvature_terms_to_use = ['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']
if use_additional_behaviors:
    curvature_terms_to_use = curvature_terms_to_use[:2]
    curvature_terms_to_use.extend(['dorsal_only_body_curvature', 'dorsal_only_head_curvature',
                                       'ventral_only_body_curvature', 'ventral_only_head_curvature',
                                       #'speed',
                                       'self_collision'])
curvature_terms_to_use


# Allow gating based on the global component
x = Xy['VB02_manifold']
x = (x-x.mean()) / x.std()  # z-score

# Just predict the residual
y = Xy['VB02'] - Xy['VB02_manifold']
y = (y-y.mean()) / y.std()  # z-score

# Interesting covariate
# curvature = Xy['vb02_curvature'][ind_data].values
curvature = Xy[curvature_terms_to_use]
curvature = (curvature-curvature.mean()) / curvature.std()  # z-score

# Package as dataframe again, and drop na values
df_model = pd.concat([pd.DataFrame({'y': y, 'x': x, 'dataset_name': Xy['dataset_name']}), 
                      pd.DataFrame(curvature)], axis=1)
df_model = df_model.dropna()
df_model.shape


# px.line(df_model.reset_index(), y=['x', 'y', 'eigenworm0'])


# df_model[['eigenworm0', 'eigenworm1', 'eigenworm2']].shape, df_model['x'].shape, df_model['y'].shape



dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()


curvature_terms_to_use


coords = {'dataset_name': dataset_name_values}

with pm.Model(coords=coords) as model:
    # Only random effect: intercept
    # hyper_intercept = pm.Normal('hyper_intercept', mu=0, sigma=1)
    # intercept = pm.Normal('intercept', mu=hyper_intercept, sigma=1, dims='dataset_name')[dataset_name_idx]
    intercept, sigma = build_baseline_priors(dims='dataset_name', dataset_name_idx=dataset_name_idx)

    # First try: pooling for sigmoid term
    x = df_model['x'].values
    sigmoid_term = build_sigmoid_term(x)

    # Also pool curvature
    curvature = df_model[curvature_terms_to_use].values
    curvature_term = build_curvature_term(curvature, curvature_terms_to_use=curvature_terms_to_use, 
                                          dims='dataset_name', dataset_name_idx=dataset_name_idx, DEBUG=True)
    
    # Expected value of outcome
    # mu = pm.Deterministic('mu', intercept[dataset_name_idx] + pm.math.dot(sigmoid_term, curvature_term))
    # product = pm.Deterministic('product', sigmoid_term*curvature_term)#pm.math.prod([sigmoid_term, curvature_term]))
    mu = pm.Deterministic('mu', intercept + sigmoid_term*curvature_term)
    # mu = pm.Deterministic('mu', intercept[dataset_name_idx] + sigmoid_term*curvature_term)
    # print(mu.shape.eval())
    # print(sigmoid_term.shape.eval())
    # # print(product.shape.eval())
    # print(intercept.shape.eval())
    # print(intercept[dataset_name_idx].shape.eval())
    # print(curvature_term.shape.eval())

    # Likelihood
    # sigma = pm.HalfCauchy("sigma", beta=0.02)
    
    y = df_model['y'].values
    likelihood = build_final_likelihood(mu, sigma=sigma, y=y, nu=100)
    # likelihood = pm.Normal('y', mu, sigma=1, observed=y)


pm.model_to_graphviz(model)


# Sample from the prior for debugging
rng = 4242
with model:
    idata = pm.sample_prior_predictive(samples=50, random_seed=rng, 
                                       var_names=['y', 'sigmoid_term','curvature_term', 'intercept', 'mu', 'sigma'])


from wbfm.utils.external.utils_arviz import plot_ts, plot_model_elements
# fig = plot_ts(idata=idata, y='y', y_hat='y')#, plot_dim='y_dim_0')
fig = plot_model_elements(idata, y_list = ['y', 'sigmoid_term', 'curvature_term', 'mu'])


# az.plot_forest(az.convert_to_dataset(idata, group='prior'), var_names=['sigma'])


# ## ACTUALLY RUN

# Run inference
with model:
    trace = pm.sample(1000, tune=1000, cores=32, chains=2)
    # trace = pm.sample(1000, tune=1000, cores=56, nuts_sampler="blackjax")


# Plot posterior distributions
az.plot_trace(trace, compact=True, var_names=['intercept', 'sigmoid_slope', 'inflection_point', 
                                              'amplitude', 'phase_shift', 'eigenworm3_coefficient']);


import matplotlib.pyplot as plt
# Generate posterior predictive samples
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['y', 'sigmoid_term'])


from wbfm.utils.external.utils_arviz import plot_ts, plot_model_elements
# fig = plot_ts(idata=idata, y='y', y_hat='y')#, plot_dim='y_dim_0')
fig = plot_model_elements(posterior_predictive, y_list = ['y', 'sigmoid_term'])


from wbfm.utils.external.utils_arviz import plot_ts
# x = np.linspace(0, posterior_predictive.observed_data['y'].shape)
# az.plot_lm(idata=posterior_predictive, y='y', x=None)
fig = plot_ts(idata=posterior_predictive, y='y', y_hat='y')#, plot_dim='y_dim_0')
# posterior_predictive.posterior_predictive.y


# # Run with functions (single dataset)

# ## Now do the comparison across other model types with my function

from wbfm.utils.external.utils_pymc import fit_multiple_models
Xy['dataset_name'].unique()


df_compare, all_traces, all_models = fit_multiple_models(Xy, 'VB02', 
                                                         # dataset_name='ZIM2165_Gcamp7b_worm1-2022_11_28',
                                                         dataset_name='2022-11-23_worm11',
                                                        use_additional_behaviors=True)
az.plot_compare(df_compare, insample_dev=False);


df_compare


all_traces['hierarchical_pca'].posterior


# Plot posterior distributions
var_names = ['intercept', 'inflection_point', 'amplitude', 
             'phase_shift', 'eigenworm3_coefficient']
az.plot_trace(all_traces['hierarchical_pca'], compact=True, var_names=var_names);


# # Plot posterior distributions
# var_names = ['intercept', #'inflection_point', #'amplitude', 
#              'phase_shift', #'eigenworm',
#              'ventral', 'dorsal', 'self']
# az.plot_forest(all_traces['hierarchical_pca'], var_names=var_names, filter_vars='like',
#               kind='ridgeplot', combined=True);


from wbfm.utils.external.utils_arviz import plot_ts, plot_model_elements
# The trace should have samples already
fig = plot_model_elements(all_traces['hierarchical_pca'], y_list = ['y', 'sigmoid_term', 'curvature_term'])


from wbfm.utils.external.utils_arviz import plot_ts
fig = plot_ts(all_traces['hierarchical_pca'])


az.plot_loo_pit(all_traces['hierarchical_pca'], y='y')


# az.plot_loo_pit(all_traces['nonhierarchical'], y='y')


# az.plot_loo_pit(all_traces['null'], y='y')


# # Now do the comparison across other model types with my function

from wbfm.utils.external.utils_pymc import fit_multiple_models


# One dataset
df_compare, all_traces, all_models = fit_multiple_models(Xy, 'VB02', dataset_name='all',
                                                        use_additional_behaviors=True)
az.plot_compare(df_compare, insample_dev=False);


# Xy['dataset_name'].unique()


df_compare, all_traces, all_models = fit_multiple_models(Xy, 'BAGL', dataset_name='ZIM2165_Gcamp7b_worm1-2022_11_28',
                                                        use_additional_behaviors=True)
az.plot_compare(df_compare, insample_dev=False);


df_compare


all_traces['hierarchical_pca']


# Plot posterior distributions
var_names = [#'intercept', 'inflection_point', 
    'amplitude', 
             'phase_shift', 'eigenworm3_coefficient', 'self_collision_coefficient']
az.plot_trace(all_traces['hierarchical_pca'], compact=True, var_names=var_names);


# # Plot posterior distributions
# var_names = ['intercept', #'inflection_point', #'amplitude', 
#              'phase_shift', #'eigenworm',
#              'ventral', 'dorsal', 'self']
# az.plot_forest(all_traces['hierarchical_pca'], var_names=var_names, filter_vars='like',
#               kind='ridgeplot', combined=True);


from wbfm.utils.external.utils_arviz import plot_ts, plot_model_elements
# The trace should have samples already
fig = plot_model_elements(all_traces['hierarchical_pca'], y_list = ['y', 'sigmoid_term', 'curvature_term'])


from wbfm.utils.external.utils_arviz import plot_ts
fig = plot_ts(all_traces['hierarchical_pca'])


# import matplotlib.pyplot as plt
# # Generate posterior predictive samples
# with all_models['hierarchical']:
#     posterior_predictive = pm.sample_posterior_predictive(all_traces['hierarchical'], var_names=['y', 'sigmoid_term'])


# from wbfm.utils.external.utils_arviz import plot_ts, plot_model_elements
# fig = plot_model_elements(posterior_predictive, y_list = ['y', 'sigmoid_term'])


# # Same thing, but with a gfp dataset

Xy_gfp['dataset_name'].unique()


df_compare_gfp, all_traces_gfp, all_models_gfp = fit_multiple_models(Xy_gfp, 'VB02', dataset_name='ZIM2319_GFP_worm3-2022-12-10',
                                                        use_additional_behaviors=True)
az.plot_compare(df_compare_gfp, insample_dev=False);


df_compare


# loo = az.loo(all_traces_gfp['hierarchical_pca'], pointwise=True)
# az.plot_khat(loo)


from wbfm.utils.external.utils_arviz import plot_ts, plot_model_elements
# The trace should have samples already
fig = plot_model_elements(all_traces_gfp['hierarchical_pca'], y_list = ['y', 'sigmoid_term', 'curvature_term', 'mu'])


from wbfm.utils.external.utils_arviz import plot_ts
fig = plot_ts(all_traces_gfp)


# az.plot_elpd(all_traces_gfp)


# Plot posterior distributions
var_names = ['intercept', 'inflection_point', #'amplitude', 
             #'phase_shift', 
             'eigenworm',
             'self', 'speed']
az.plot_forest(all_traces['hierarchical_pca'], var_names=var_names, filter_vars='like',
              kind='ridgeplot', combined=True);


all_traces_gfp['hierarchical_pca']





# ## Many neurons

# One dataset
df_compare, all_traces, all_models = fit_multiple_models(Xy, 'VB02', dataset_name='all')
az.plot_compare(df_compare, insample_dev=False);


neuron_list = ['VB02', 'DB01', 'DB02', 'VB01', 'VB03', 'RMEV', 'URXL', 'BAGL', 'AVAL', 'ALA', 'SMDDL']
all_df_compare = {}
all_traces = {}
all_models = {}

for neuron in neuron_list:

    _df_compare, _all_traces, _all_models = fit_multiple_models(Xy, neuron, dataset_name='all')
    az.plot_compare(_df_compare, insample_dev=False);
    plt.title(neuron)
    plt.show()

    all_df_compare[neuron] = _df_compare
    all_traces[neuron] = _all_traces
    all_models[neuron] = _all_models


# df_compare_db, all_traces_db, all_models_db = fit_multiple_models(Xy, 'DB01', dataset='all')
# az.plot_compare(df_compare_db, insample_dev=False);


# # Load from disk and plot

from wbfm.utils.external.utils_arviz import plot_model_elements
from wbfm.utils.external.utils_pymc import get_dataframe_for_single_neuron, build_baseline_priors, build_sigmoid_term, build_curvature_term, build_final_likelihood, build_sigmoid_term_pca


fname = "/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/output/VB02_hierarchical_pca_trace.nc"
trace = az.from_netcdf(fname)


# # # Plot posterior distributions
# # az.plot_trace(trace, compact=True, var_names=['intercept', 'sigmoid_slope', 'inflection_point', 
# #                                               'amplitude', 'phase_shift', 'eigenworm3_coefficient']);
# x = trace.posterior.phase_shift
# y = trace.posterior.eigenworm3_coefficient
# az.plot_kde(x, y, contourf_kwargs={"cmap": "Blues"})
# # az.plot_pair(trace, var_names=['sigmoid_slope', 'inflection_point', 'phase_shift', 'eigenworm3_coefficient'])


fig_raw = plot_model_elements(trace, to_show=False,
                         y_list=['sigmoid_term', 'curvature_term'])

fig_raw.update_xaxes(range=Xy_ind_range.iloc[12, :])


# Redefine model
# First pack into a single dataframe for preprocessing, then unpack
neuron_name = 'VB02'

df_model = get_dataframe_for_single_neuron(Xy, neuron_name)
x = df_model['x'].values
y = df_model['y'].values
curvature = df_model[['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']].values
pca_modes = df_model[['x_pca0', 'x_pca1']].values

dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
dims = 'dataset_name'
dim_opt = dict(dims=dims, dataset_name_idx=dataset_name_idx)

dataset_name_idx, dataset_name_values = df_model.dataset_name.factorize()
coords = {'dataset_name': dataset_name_values}
dims = 'dataset_name'
with pm.Model(coords=coords) as hierarchical_model:
    # Full model
    intercept, sigma = build_baseline_priors()#**dim_opt)
    sigmoid_term = build_sigmoid_term_pca(pca_modes, **dim_opt)
    curvature_term = build_curvature_term(curvature, **dim_opt)

    mu = pm.Deterministic('mu', intercept + sigmoid_term * curvature_term)
    likelihood = build_final_likelihood(mu, sigma, y)


with hierarchical_model:
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['y', 'sigmoid_term', 'curvature_term', 'mu'])


fig = plot_model_elements(posterior_predictive, to_show=False, y_list=['y', 'sigmoid_term', 'curvature_term'])


import kaleido


# from wbfm.utils.general.utils_paper import apply_figure_settings
# Get the indices for each dataset
Xy_ind = Xy.reset_index().groupby('dataset_name')['index']
Xy_ind_range = pd.DataFrame({'min': Xy_ind.min(), 'max': Xy_ind.max()})

# name = '2022-11-23_worm9'
idx = 12

fig.update_xaxes(range=Xy_ind_range.iloc[idx, :])
fig.update_yaxes(range=[-2, 3])
# fig.update_traces(title=f'Model for VB02 in dataset {Xy_ind_range.index[idx]}')
# apply_figure_settings(fig, height_factor=0.2)

fig.show()

fname = os.path.join("../../paper/multiplexing", f'example_vb02_model_{Xy_ind_range.index[idx]}.png')
fig.write_image(fname, scale=3, engine='kaleido')
# fname = Path(fname).with_suffix('.svg')
# fig.write_image(fname)


Xy_ind_range.index[12], Xy_ind_range.iloc[12, :]


# ## Save for further plotting

fname = "/scratch/neurobiology/zimmer/fieseler/paper/hierarchical_modeling/output/VB02_hierarchical_pca_pps.nc"
posterior_predictive.to_netcdf(fname)


# ## Run for all neurons

from wbfm.utils.external.utils_arviz import load_from_disk_and_plot
from pathlib import Path
for fname in Path(os.path.join(get_hierarchical_modeling_dir(), 'output')).iterdir():
    if fname.name.endswith('.nc'):
        try:
            fig = load_from_disk_and_plot(str(fname.resolve()), update_original_trace=True, verbose=1)
        except (KeyError, OSError):
            continue

        # Clear traces so that it doesn't keep using tons of memory
        if fig is not None:run
            fig.data = []
        # if fig is not None:
        #     break


# # Test with alternative models

df_compare2, all_traces2, all_models2 = fit_multiple_models(Xy, 'VB02', 
                                                         # dataset_name='ZIM2165_Gcamp7b_worm1-2022_11_28',
                                                         dataset_name='2022-11-23_worm11',
                                                        use_additional_behaviors=True)
az.plot_compare(df_compare2, insample_dev=False);


from wbfm.utils.external.utils_arviz import plot_ts, plot_model_elements
# The trace should have samples already
fig = plot_model_elements(all_traces2['hierarchical_pca'], y_list = ['y', 'sigmoid_term', 'curvature_term'])
# fig = plot_model_elements(all_traces2['nonhierarchical'], y_list = ['y', 'curvature_term'])


# all_traces2['hierarchical_pca_model_with_drift']


fig = plt.figure(figsize=(8, 6), constrained_layout=False)
ax = fig.add_subplot(111, xlabel="time", )#ylabel="beta", title="Change of beta over time")
ax.plot(az.extract(all_traces2['hierarchical_pca_model_with_drift'], var_names="alpha"), "b", alpha=0.05);

# ax.xaxis.set_major_locator(ticks_changes)
# ax.set_xticklabels(ticklabels_changes)


all_traces2['hierarchical_pca_model_with_drift']


from wbfm.utils.external.utils_arviz import plot_ts
# x = np.linspace(0, posterior_predictive.observed_data['y'].shape)
# az.plot_lm(idata=posterior_predictive, y='y', x=None)
fig = plot_ts(all_traces2['hierarchical_pca_model_with_drift'], y='y', y_hat='y')#, plot_dim='y_dim_0')
# posterior_predictive.posterior_predictive.y




