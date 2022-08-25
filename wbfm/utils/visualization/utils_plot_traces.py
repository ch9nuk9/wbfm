import numpy as np
import pandas as pd
from lmfit.models import ExponentialModel, ConstantModel
from sklearn.preprocessing import StandardScaler
import scipy


def build_trace_factory(base_trace_fname, trace_mode, smoothing_func=lambda x: x, background_per_pixel=0):
    if trace_mode in ['red', 'green']:
        fname = base_trace_fname.with_name(f"{trace_mode}_traces.h5")
        df = pd.read_hdf(fname)
        neuron_names = list(set(df.columns.get_level_values(0)))

        def get_y_raw(i):
            y_raw = df[i]['brightness']
            return smoothing_func(y_raw - background_per_pixel * df[i]['volume'])

    else:
        fname = base_trace_fname.with_name("red_traces.h5")
        df_red = pd.read_hdf(fname)
        fname = base_trace_fname.with_name("green_traces.h5")
        df_green = pd.read_hdf(fname)
        neuron_names = list(set(df_green.columns.get_level_values(0)))

        def get_y_raw(i):
            red_raw = df_red[i]['brightness']
            green_raw = df_green[i]['brightness']
            vol = df_green[i]['volume']  # Same for both
            return smoothing_func((green_raw - vol * background_per_pixel) / (red_raw - vol * background_per_pixel))
    print(f"Read traces from: {fname}")

    return get_y_raw, neuron_names


def check_default_names(all_names, num_neurons):
    if all_names is None:
        all_names = [str(i) for i in range(num_neurons)]
    return all_names


def set_big_font(size=22):
    # From: https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    import matplotlib

    font = {'weight': 'bold',
            'size': size}
    matplotlib.rc('font', **font)


def detrend_exponential(y_with_nan):
    """
    Bleach correction via simple exponential fit, subtraction, and re-adding the mean

    Uses np.polyfit on np.log(y), with errors weighted back to the data space. See:

    https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly

    Parameters
    ----------
    y_with_nan

    Returns
    -------

    """

    ind = np.where(~np.isnan(y_with_nan))[0]
    t = np.squeeze(StandardScaler(copy=False).fit_transform(ind.reshape(-1, 1)))
    y = y_with_nan[ind]
    y_log = np.log(y)

    fit_vars = np.polyfit(t, y_log, 1)#, w=np.sqrt(y))

    # Subtract in the original data space
    y_fit = np.exp(fit_vars[0]) * np.exp(t*fit_vars[1])
    y_corrected = y - y_fit + np.mean(y)

    return ind, y_corrected


def detrend_exponential_lmfit(y_with_nan, x=None):
    """
    Bleach correction via simple exponential fit, subtraction, and re-adding the mean

    Uses np.polyfit on np.log(y), with errors weighted back to the data space. See:

    https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly

    Parameters
    ----------
    x
    y_with_nan

    Returns
    -------

    """

    exp_mod = ExponentialModel(prefix='exp_')
    ind = np.where(~np.isnan(y_with_nan))[0]
    if x is None:
        x = ind
    else:
        x = x[ind]
    y = y_with_nan[ind]
    out = None

    try:
        pars = exp_mod.guess(y, x=x)
        out = exp_mod.fit(y, pars, x=x)

        comps = out.eval_components(x=x)
        y_fit = comps['exp_']
        y_corrected = y / y_fit

        y_corrected_with_nan = np.empty_like(y_with_nan)
        y_corrected_with_nan[:] = np.nan
        y_corrected_with_nan[ind] = y_corrected
        flag = True

    except TypeError:
        # Occurs when there are too few input points
        y_corrected_with_nan, y_fit = y_with_nan, y_with_nan
        flag = False

    if out is None or not out.errorbars or 0 in y_fit:
        # Crude measurement of bad convergence, even if it didn't error out
        y_corrected_with_nan, y_fit = y_with_nan, y_with_nan
        flag = False

    return y_corrected_with_nan, (y_fit, out, flag)


def detrend_exponential_lmfit_give_indices(y_full, ind_iter):
    y = y_full[ind_iter]
    ind_remove_nan = np.where(~np.isnan(y_full))[0]
    y_no_nan = y_full[ind_remove_nan]
    x = ind_iter

    exp_mod = ExponentialModel(prefix='exp_')
    out = None

    try:
        pars = exp_mod.guess(y, x=x)
        out = exp_mod.fit(y, pars, x=x)

        comps = out.eval_components(x=ind_remove_nan)
        y_fit = comps['exp_']
        y_corrected = y_no_nan / y_fit

        y_corrected_with_nan = np.empty_like(y_full)
        y_corrected_with_nan[:] = np.nan
        y_corrected_with_nan[ind_remove_nan] = y_corrected
    except TypeError:
        # Occurs when there are too few input points
        y_corrected_with_nan, y_fit = y_full, y_full

    return y_corrected_with_nan, (y_fit, out)


def detrend_exponential_iter(trace, max_iters=100, convergence_threshold=0.01,
                             low_quantile=0.15, high_quantile=0.85):
    """
    low/high_quantile: how many percent of the data should be excluded at bottom/top

    Parameters
    ----------
    trace
    convergence_threshold - stop if L2 norm changes by less than this
    low_quantile - per iteration, remove this bottom percentile
    high_quantile - per iteration, remove this bottom percentile

    Returns
    -------

    """
    y_full = trace
    ind_iter = np.where(~np.isnan(y_full))[0]
    y_fit = np.array([0]*len(ind_iter))

    for num_iter in range(max_iters):
        y_detrend = detrend_exponential_lmfit_give_indices(y_full, ind_iter)[0]
        y_fit_last = y_fit
        y_fit = detrend_exponential_lmfit_give_indices(y_full, ind_iter)[1][0]
        ind_iter = np.where(np.logical_and(np.nanquantile(y_detrend, low_quantile) < y_detrend, y_detrend < np.nanquantile(y_detrend,high_quantile)))[0]
        if scipy.spatial.distance.euclidean(y_fit, y_fit_last) <= convergence_threshold:
            break
    return y_detrend, num_iter
