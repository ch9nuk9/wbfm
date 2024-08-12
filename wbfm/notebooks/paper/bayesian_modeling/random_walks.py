#!/usr/bin/env python
# coding: utf-8

import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings(action="ignore", category=MatplotlibDeprecationWarning)


RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
az.style.use("arviz-darkgrid")


# from pandas_datareader import data
# prices = data.GoogleDailyReader(symbols=['GLD', 'GFI'], end='2014-8-1').read().loc['Open', :, :]
try:
    prices = pd.read_csv(os.path.join("..", "data", "stock_prices.csv")).dropna()
except FileNotFoundError:
    prices = pd.read_csv(pm.get_data("stock_prices.csv")).dropna()

prices["Date"] = pd.DatetimeIndex(prices["Date"])
prices = prices.set_index("Date")
prices_zscored = (prices - prices.mean()) / prices.std()
prices.head()


# # Random walk model

with pm.Model(coords={"time": prices.index.values}) as model_randomwalk:
    # std of random walk
    sigma_alpha = pm.Exponential("sigma_alpha", 50.0)
    sigma_beta = pm.Exponential("sigma_beta", 50.0)

    alpha = pm.GaussianRandomWalk(
        "alpha", sigma=sigma_alpha, init_dist=pm.Normal.dist(0, 10), dims="time"
    )
    beta = pm.GaussianRandomWalk(
        "beta", sigma=sigma_beta, init_dist=pm.Normal.dist(0, 10), dims="time"
    )


with model_randomwalk:
    # Define regression
    regression = alpha + beta * prices_zscored.GFI.values

    # Assume prices are Normally distributed, the mean comes from the regression.
    sd = pm.HalfNormal("sd", sigma=0.1)
    likelihood = pm.Normal("y", mu=regression, sigma=sd, observed=prices_zscored.GLD.to_numpy())


with model_randomwalk:
    trace_rw = pm.sample(tune=2000, target_accept=0.9)


# # Plots

fig = plt.figure(figsize=(8, 6), constrained_layout=False)
ax = plt.subplot(111, xlabel="time", ylabel="alpha", title="Change of alpha over time.")
ax.plot(az.extract(trace_rw, var_names="alpha"), "r", alpha=0.05)

ticks_changes = mticker.FixedLocator(ax.get_xticks().tolist())
ticklabels_changes = [str(p.date()) for p in prices[:: len(prices) // 7].index]
ax.xaxis.set_major_locator(ticks_changes)
ax.set_xticklabels(ticklabels_changes)

fig.autofmt_xdate()


fig = plt.figure(figsize=(8, 6), constrained_layout=False)
ax = fig.add_subplot(111, xlabel="time", ylabel="beta", title="Change of beta over time")
ax.plot(az.extract(trace_rw, var_names="beta"), "b", alpha=0.05)

ax.xaxis.set_major_locator(ticks_changes)
ax.set_xticklabels(ticklabels_changes)

fig.autofmt_xdate()


trace_rw


# # Prior plots


time = np.arange(1600)
with pm.Model(coords={"time": time}) as model_randomwalk:
    # std of random walk
    # sigma_alpha = pm.Exponential("sigma_alpha", 1000.0)
    alpha = pm.GaussianRandomWalk(
        "alpha", sigma=0.01, init_dist=pm.Normal.dist(0, 1.0), dims="time"
    )
    # Define just random walk
    regression = alpha

    # Assume prices are Normally distributed, the mean comes from the regression.
    sd = pm.HalfNormal("sd", sigma=0.01)
    likelihood = pm.Normal("y", mu=regression, sigma=sd, observed=time)

    # Just sample from the prior
    idata = pm.sample_prior_predictive(samples=10, random_seed=rng)


_, ax = plt.subplots()

x = time #xr.DataArray(np.linspace(-2, 2, 50), dims=["plot_dim"])
y = idata.prior_predictive

ax.plot(x, y.stack(sample=("chain", "draw")).to_dataarray().squeeze(), c="k", alpha=0.4)

ax.set_xlabel("Predictor (stdz)")
ax.set_ylabel("Mean Outcome (stdz)")
ax.set_title("Prior predictive checks -- Random walk");


az.plot_ppc(idata, group='prior')#, var_names=['sigma_alpha'])


# data = az.load_arviz_data('radon')
# data


with pm.Model() as model_1:
    a = pm.Normal("a", 0.0, 0.5)
    b = pm.Normal("b", 0.0, 1.0)

    mu = a + b * predictor_scaled
    sigma = pm.Exponential("sigma", 1.0)

    pm.Normal("obs", mu=mu, sigma=sigma, observed=outcome_scaled)
    idata = pm.sample_prior_predictive(draws=50, random_seed=rng)




