#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import pandas as pd
import plotly.express as px
import bambi as bmb
import arviz as az
import numpy as np
fname = 'data.h5'
Xy = pd.read_hdf(fname)
import pymc as pm


# In[2]:


Xy.head()


# # Look at one dataset

# In[3]:


# Choose one dataset
ind_data = Xy['dataset_name'] == '2022-11-23_worm8'
# Also remove outliers
# ind_inliers = Xy.loc[ind, 'VB02'].abs() < 0.15
px.scatter(Xy[ind_data], x='vb02_curvature', y='VB02', color='fwd', trendline='ols')


# In[60]:


Xy[ind_data]


# # Baseline: on one dataset, use the vb02 curvature directly

# In[4]:


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


# In[5]:


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


# In[6]:


pm.model_to_graphviz(model)


# In[7]:


# Run inference
with model:
    trace = pm.sample(1000, tune=1000, cores=64)


# In[8]:


# Plot posterior distributions
az.plot_trace(trace, compact=True, var_names=['sigmoid_slope', 'inflection_point', 'amplitude']);


# In[9]:


import matplotlib.pyplot as plt
# Generate posterior predictive samples
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['y', 'sigmoid_term'])


# In[10]:


# Average across chains, but not draws
y_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['y'], axis=0), axis=0)
sigmoid_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['sigmoid_term'], axis=0), axis=0)


# In[11]:


df_pred = pd.DataFrame({'y': y, 'y_pred': y_pred, 'curvature': curvature, 'manifold': x,
                        'fwd': Xy['fwd'][ind_data], 'sigmoid_pred': sigmoid_pred})
px.line(df_pred)


# # Baseline 2: on one dataset, use the eigenworms

# In[12]:



ind_data = Xy['dataset_name'] == '2022-11-23_worm8'

# Allow gating based on the global component
x = Xy['VB02_manifold'][ind_data].values
x = (x-x.mean()) / x.std()  # z-score

# Just predict the residual
y = Xy['VB02'][ind_data].values - Xy['VB02_manifold'][ind_data].values
y = (y-y.mean()) / y.std()  # z-score

# Interesting covariate
# curvature = Xy['vb02_curvature'][ind_data].values
curvature = Xy[['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']][ind_data].values
curvature = (curvature-curvature.mean()) / curvature.std()  # z-score

print(x.shape, y.shape, curvature.shape)


# In[13]:


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

    # New: build a linear combination of the eigenworms as the covariate that is rectified
    # New: constrain the coefficients to sum to 1, preventing cross-talk with the amplitude
    # coefficients_vec = pm.Normal('eigenworm_coefficient', mu=0, sigma=1, shape=curvature.shape[1])
    # coefficients_vec = pm.Dirichlet('eigenworm_coefficient', a=np.ones(curvature.shape[1]))
    
    # Alternative: Define covariance matrix to enforce sum of squares constraint
    num_coefficients = curvature.shape[1]
    covariance_matrix = np.eye(num_coefficients) / num_coefficients
    coefficients_vec = pm.MvNormal('coefficients_vec', mu=0, cov=covariance_matrix, shape=num_coefficients)
    
    curvature_term = pm.Deterministic('curvature_term', pm.math.dot(curvature, coefficients_vec))
    
    # Expected value of outcome
    # mu = intercept + amplitude * sigmoid_term * curvature_term
    mu = pm.Deterministic('mu', intercept + amplitude * sigmoid_term * curvature_term)

    # Likelihood
    sigma = pm.HalfNormal('sigma', sigma=1)
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)


# In[14]:



for rv, shape in model.eval_rv_shapes().items():
    print(f"{rv:>11}: shape={shape}")


# In[15]:


pm.model_to_graphviz(model)


# In[16]:


# Run inference
with model:
    trace = pm.sample(1000, tune=1000, cores=64)


# In[17]:


# Plot posterior distributions
az.plot_trace(trace, compact=True, var_names=['sigmoid_slope', 'inflection_point', 'amplitude', 'coefficients_vec']);


# In[18]:


import matplotlib.pyplot as plt
# Generate posterior predictive samples
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['y', 'sigmoid_term'])


# In[32]:


az.plot_pair(trace, var_names=['coefficients_vec'])


# In[26]:


# # Average across chains, but not draws
# y_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['y'], axis=0), axis=0)
# sigmoid_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['sigmoid_term'], axis=0), axis=0)


# In[62]:


# df_pred = pd.DataFrame({'y': y, 'y_pred': y_pred, #'curvature': curvature, 
#                         'manifold': x,
#                         'fwd': Xy['fwd'][ind_data], 'sigmoid_pred': sigmoid_pred})
# px.line(df_pred)


# In[25]:


from wbfm.utils.external.utils_arviz import plot_ts
# x = np.linspace(0, posterior_predictive.observed_data['y'].shape)
# az.plot_lm(idata=posterior_predictive, y='y', x=None)
fig = plot_ts(idata=posterior_predictive, y='y', y_hat='y')#, plot_dim='y_dim_0')
# posterior_predictive.posterior_predictive.y


# In[ ]:





# # Baseline 3: use the eigenworms on a neuron which shouldn't show anything (ALA)

# In[33]:



ind_data = Xy['dataset_name'] == '2022-11-23_worm8'

# Allow gating based on the global component
x = Xy['ALA_manifold'][ind_data].values
x = (x-x.mean()) / x.std()  # z-score

# Just predict the residual
y = Xy['ALA'][ind_data].values - Xy['ALA_manifold'][ind_data].values
y = (y-y.mean()) / y.std()  # z-score

# Interesting covariate
# curvature = Xy['vb02_curvature'][ind_data].values
curvature = Xy[['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']][ind_data].values
curvature = (curvature-curvature.mean()) / curvature.std()  # z-score

print(x.shape, y.shape, curvature.shape)


# In[76]:


def add_baseline(model):
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    return intercept, sigma

with pm.Model() as model:
    # Priors for parameters
    intercept, sigma = add_baseline(model)
    # intercept = pm.Normal('intercept', mu=0, sigma=10)
    # sigma = pm.HalfNormal('sigma', sigma=1)
    # Sigmoid (hierarchy) term
    log_sigmoid_slope = pm.Normal('log_sigmoid_slope', mu=0, sigma=1)  # Using log-amplitude for positivity
    inflection_point = pm.Normal('inflection_point', mu=0, sigma=2)
    log_amplitude = pm.Normal('log_amplitude', mu=0, sigma=2)  # Using log-amplitude for positivity
    # Baseline term
 
    # Transforming log-amplitude to ensure positivity
    amplitude = pm.Deterministic('amplitude', pm.math.exp(log_amplitude))
    sigmoid_slope = pm.Deterministic('sigmoid_slope', pm.math.exp(log_sigmoid_slope))

    # Sigmoid term
    sigmoid_term = pm.Deterministic('sigmoid_term', pm.math.sigmoid(sigmoid_slope * (x - inflection_point)))

    # New: build a linear combination of the eigenworms as the covariate that is rectified
    # New: constrain the coefficients to sum to 1, preventing cross-talk with the amplitude
    # coefficients_vec = pm.Normal('eigenworm_coefficient', mu=0, sigma=1, shape=curvature.shape[1])
    # coefficients_vec = pm.Dirichlet('eigenworm_coefficient', a=np.ones(curvature.shape[1]))
    
    # Alternative: Define covariance matrix to enforce sum of squares constraint
    num_coefficients = curvature.shape[1]
    covariance_matrix = np.eye(num_coefficients) / num_coefficients
    coefficients_vec = pm.MvNormal('coefficients_vec', mu=0, cov=covariance_matrix, shape=num_coefficients)
    
    curvature_term = pm.Deterministic('curvature_term', pm.math.dot(curvature, coefficients_vec))
    
    # Expected value of outcome
    # mu = intercept + amplitude * sigmoid_term * curvature_term
    mu = pm.Deterministic('mu', intercept + amplitude * sigmoid_term * curvature_term)

    # Likelihood
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)


# In[77]:


# model.to_graphviz()


# In[78]:



# for rv, shape in model.eval_rv_shapes().items():
#     print(f"{rv:>11}: shape={shape}")


# In[79]:


# Run inference
with model:
    trace = pm.sample(1000, tune=1000, cores=64)


# In[38]:


# Plot posterior distributions
az.plot_trace(trace, compact=True, var_names=['sigmoid_slope', 'inflection_point', 'amplitude', 'coefficients_vec']);


# In[39]:


import matplotlib.pyplot as plt
# Generate posterior predictive samples
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['y', 'sigmoid_term'])


# In[40]:


# Average across chains, but not draws
y_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['y'], axis=0), axis=0)
sigmoid_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['sigmoid_term'], axis=0), axis=0)


# In[41]:


df_pred = pd.DataFrame({'y': y, 'y_pred': y_pred, #'curvature': curvature, 
                        'manifold': x,
                        'fwd': Xy['fwd'][ind_data], 'sigmoid_pred': sigmoid_pred})
px.line(df_pred)


# In[47]:


from wbfm.utils.external.utils_arviz import plot_ts
# x = np.linspace(0, posterior_predictive.observed_data['y'].shape)
# az.plot_lm(idata=posterior_predictive, y='y', x=None)
fig = plot_ts(idata=posterior_predictive, y='y', y_hat='y', num_samples=1000)#, plot_dim='y_dim_0')
# posterior_predictive.posterior_predictive.y


# In[46]:


with model:
    pm.compute_log_likelihood(trace)
az.loo(trace)


# # Baseline 4: use the eigenworms on a neuron which shouldn't show anything (AVA)

# In[48]:



ind_data = Xy['dataset_name'] == '2022-11-23_worm8'

# Allow gating based on the global component
x = Xy['AVAL_manifold'][ind_data].values
x = (x-x.mean()) / x.std()  # z-score

# Just predict the residual
y = Xy['AVAL'][ind_data].values - Xy['AVAL_manifold'][ind_data].values
y = (y-y.mean()) / y.std()  # z-score

# Interesting covariate
# curvature = Xy['vb02_curvature'][ind_data].values
curvature = Xy[['eigenworm0', 'eigenworm1', 'eigenworm2', 'eigenworm3']][ind_data].values
curvature = (curvature-curvature.mean()) / curvature.std()  # z-score

print(x.shape, y.shape, curvature.shape)


# In[49]:


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

    # New: build a linear combination of the eigenworms as the covariate that is rectified
    # New: constrain the coefficients to sum to 1, preventing cross-talk with the amplitude
    # coefficients_vec = pm.Normal('eigenworm_coefficient', mu=0, sigma=1, shape=curvature.shape[1])
    # coefficients_vec = pm.Dirichlet('eigenworm_coefficient', a=np.ones(curvature.shape[1]))
    
    # Alternative: Define covariance matrix to enforce sum of squares constraint
    num_coefficients = curvature.shape[1]
    covariance_matrix = np.eye(num_coefficients) / num_coefficients
    coefficients_vec = pm.MvNormal('coefficients_vec', mu=0, cov=covariance_matrix, shape=num_coefficients)
    
    curvature_term = pm.Deterministic('curvature_term', pm.math.dot(curvature, coefficients_vec))
    
    # Expected value of outcome
    # mu = intercept + amplitude * sigmoid_term * curvature_term
    mu = pm.Deterministic('mu', intercept + amplitude * sigmoid_term * curvature_term)

    # Likelihood
    sigma = pm.HalfNormal('sigma', sigma=1)
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)


# In[50]:


# Run inference
with model:
    trace = pm.sample(1000, tune=1000, cores=64)


# In[51]:


# Plot posterior distributions
az.plot_trace(trace, compact=True, var_names=['sigmoid_slope', 'inflection_point', 'amplitude', 'coefficients_vec']);


# In[52]:


import matplotlib.pyplot as plt
# Generate posterior predictive samples
with model:
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['y', 'sigmoid_term'])


# In[53]:


# Average across chains, but not draws
y_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['y'], axis=0), axis=0)
sigmoid_pred = np.mean(np.mean(posterior_predictive.posterior_predictive['sigmoid_term'], axis=0), axis=0)


# In[54]:


df_pred = pd.DataFrame({'y': y, 'y_pred': y_pred, #'curvature': curvature, 
                        'manifold': x,
                        'fwd': Xy['fwd'][ind_data], 'sigmoid_pred': sigmoid_pred})
px.line(df_pred)


# In[55]:


from wbfm.utils.external.utils_arviz import plot_ts
# x = np.linspace(0, posterior_predictive.observed_data['y'].shape)
# az.plot_lm(idata=posterior_predictive, y='y', x=None)
fig = plot_ts(idata=posterior_predictive, y='y', y_hat='y')#, plot_dim='y_dim_0')
# posterior_predictive.posterior_predictive.y


# In[56]:


with model:
    pm.compute_log_likelihood(trace)
az.loo(trace)

