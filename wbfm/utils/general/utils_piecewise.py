import warnings

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor
from statsmodels.tools import add_constant
from tqdm.auto import tqdm

from wbfm.utils.general.high_performance_pandas import get_names_from_df

##
# Top level functions with pre-made filtering options
##


def plot_ransac_corrected_traces(x, y, ratio, vol, xlim=None, include_red=False):
    if xlim is None:
        xlim = [500, 1100]

    # Predict
    x_with_vol = pd.concat([x, vol], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        y_pred = rolling_filter_trace_using_func(y, x_with_vol, _ransac_process, 256, delta=32)
    y_corrected = y - y_pred

    # Plot
    fig, axes = plt.subplots(nrows=3, dpi=300)
    axes[0].plot(y_pred, label='predicted green', color='tab:orange')
    if include_red:
        axes[0].plot(x / x.mean() * y.mean(), label='scaled red', color='tab:red')
    axes[0].plot(y, label='green', color='tab:green')
    # plt.plot(y_corrected, label='corrected_green')
    ratio2 = y / y_pred
    axes[1].plot(ratio2, label='corrected ratio', color='tab:red')
    ratio_norm = ratio / ratio.mean() * ratio2.mean()
    axes[1].plot(ratio_norm, label='original ratio', color='tab:blue')
    # plt.xlim(1200, 1500)
    axes[1].set_ylim(0.8*np.nanquantile(ratio2, 0.05), 1.5*np.nanquantile(ratio2, 0.98))
    axes[0].set_ylim(0.8*np.nanquantile(y, 0.05), 1.5*np.nanquantile(y, 0.98))
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlim(xlim[0], xlim[1])
    axes[1].set_xlim(xlim[0], xlim[1])

    axes[2].plot(x, y, 'o')
    axes[2].set_ylabel("Green")
    axes[2].set_xlabel("Red")

    return y_corrected, y_pred


def _ransac_process(y, x, to_add_constant=True, random_state=42):
    if to_add_constant:
        predictors = add_constant(x)
    else:
        predictors = x
    reg = RANSACRegressor(random_state=random_state).fit(predictors, y)
    return reg.predict(predictors)


def predict_using_rolling_ransac_filter_single_trace(red, green, vol=None, nperseg=256):

    # Predict
    if vol is not None:
        red = pd.concat([red, vol], axis=0).copy()

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        y_pred = rolling_filter_trace_using_func(green, red, _ransac_process, nperseg, delta=32)

    return y_pred


def predict_using_rolling_ransac_with_nan(df: pd.DataFrame, target_column: str, predictor_columns: list=None,
                                          **kwargs):
    """
    Nan-safe version of predict_using_rolling_ransac_filter_single_trace with different interface to conform to pandas

    WARNING: very slow, because pandas calculates by returning scalars
    https://github.com/pandas-dev/pandas/issues/4130

    Parameters
    ----------
    df - should have all
    target_column - name of the column in the dataframe to predict
    predictor_columns - names of the columns in the dataframe to use when predicting
    kwargs

    Returns
    -------
    pandas series of the single trace, as predicted using rolling ransac

    """
    if predictor_columns is None:
        predictor_columns = [col for col in df.columns if col != target_column]

    # Pandas rolling can't do multiple columns by default, so build a closure, following:
    # https://stackoverflow.com/questions/41701246/how-to-get-rolling-pandas-dataframe-subsets
    # Note that the answer is partially wrong, and the closure must return a scalar

    def dataframe_roll(_df):
        def my_fn(window_series):
            i0, i1 = window_series.index[0], window_series.index[-1]
            window_df = _df[(_df.index >= i0) & (_df.index <= i1)]
            predictor_df = window_df[predictor_columns]
            valid_ind = window_series.dropna().index.intersection(predictor_df.dropna().index)

            _out = _ransac_process(window_series.loc[valid_ind], predictor_df.loc[valid_ind, :],
                                   to_add_constant=False)
            i_middle = int(len(_out) / 2)
            return _out[i_middle]

        return my_fn

    rol = df[target_column].rolling(**kwargs)
    closure_func = dataframe_roll(df)
    result = rol.apply(closure_func, raw=False)

    return result


def apply_rolling_wiener_filter_full_dataframe(red, green, strength='strong', nperseg=128, **kwargs):
    opt = dict(nperseg=nperseg)
    if 'factor' not in kwargs:
        if strength == 'strong':
            opt['factor'] = 3
        elif strength == 'weak':
            opt['factor'] = 4
        else:
            raise NotImplementedError(f"Unknown value: {strength}")
    delta = 32
    df_filt = apply_function_to_red_and_green_dataframes(red, green, _wiener_filter, delta=delta, window=nperseg,
                                                         **opt, **kwargs)

    return df_filt


def apply_rolling_wiener_filter_single_trace(y, y_red, strength='strong', nperseg=128):

    if strength == 'strong':
        factor = 4
    elif strength == 'weak':
        factor = 3
    elif strength == 'weaker':
        factor = 2
    elif strength == 'weakest':
        factor = 1
    else:
        raise NotImplementedError(f"Unknown value: {strength}")
    delta = 32
    opt = dict(factor=factor, nperseg=nperseg)
    y_filt = rolling_filter_trace_using_func(y, y_red, _wiener_filter, delta=delta, window=nperseg, **opt)

    return y_filt


def _wiener_filter(trace, noise, factor, nperseg):
    # scaler = StandardScaler()
    trace_norm = trace  # scaler.fit_transform(np.array(trace).reshape(-1, 1))
    trace_filtered = scipy.signal.wiener(np.squeeze(trace_norm), mysize=int(nperseg / (2 ** factor)) + 1,
                                         noise=noise)
    # return scaler.inverse_transform(trace_filtered)
    return trace_filtered

##
# User functions, but lower level
##


def apply_function_to_red_and_green_dataframes(red, green, func, **kwargs):

    idx_intersect = red.columns.intersection(green.columns)
    red = red[idx_intersect]
    green = green[idx_intersect]

    names = get_names_from_df(red)
    traces = dict()

    for name in tqdm(names, leave=False):
        y, x = green[name], red[name]
        new_trace = rolling_filter_trace_using_func(y, x, func, **kwargs)
        traces[name] = new_trace

    df_filtered = pd.DataFrame(traces)
    return df_filtered


def rolling_filter_trace_using_func(y, x, func, window, delta, filtfilt=True, **kwargs):
    """
    Applies a filter function (func) to the trace in a rolling window fashion

    Parameters
    ----------
    y - trace (target)
    x - noise or predictor
    func - function for filtering. Signature: f(y, x, **kwargs)
    window - size of rolling window
    filtfilt - to apply twice (forward and backward), and average
    kwargs

    Returns
    -------

    """
    edges = build_window_edges(len(y), window=window, delta=delta)
    filtered_fragments, noise_fragments, raw_fragments, used_edges = apply_function_to_windows(y, edges, func,
                                                                                               noise_trace=x,
                                                                                               **kwargs)
    y_filt = combine_trace_fragments(filtered_fragments, used_edges, len(y))

    if filtfilt:
        y = np.flip(y)
        if x is not None:
            x = np.flip(x)
        filtered_fragments, noise_fragments, raw_fragments, used_edges = apply_function_to_windows(y, edges, func,
                                                                                                   noise_trace=x,
                                                                                                   **kwargs)
        y_filt2 = np.flip(combine_trace_fragments(filtered_fragments, used_edges, len(y)))
        y_filt = (y_filt + y_filt2) / 2.0

    return y_filt


##
# Helpers
##


def build_window_edges(full_size, window=128, delta=None):
    if delta is None:
        delta = int(window / 2)

    starts = np.arange(0, full_size, delta)
    ends = starts + window
    return list(zip(starts, ends))


def apply_function_to_windows(trace, window_edges, func, noise_trace=None, min_pts_required=64, **kwargs):
    """

    Parameters
    ----------
    trace
    window_edges
    func - signature is f(trace, noise_trace, **kwargs) -> output trace
        Output could be a prediction or a cleaned version, or anything
    noise_trace
    min_pts_required
    kwargs

    Returns
    -------

    """
    filtered_fragments = []
    raw_fragments = []
    noise_fragments = []
    used_edges = []
    for edges in window_edges:
        i0, i1 = edges
        if i1 > len(trace):
            pad_len = i1 - len(trace)
            num_real_pts = i1 - i0 - pad_len
            if num_real_pts < min_pts_required:
                # print("Skipping window; too few points")
                continue
            this_trace = np.pad(trace, pad_len)
            if noise_trace is not None:
                this_noise_trace = np.pad(noise_trace, pad_len)
        else:
            this_trace = trace
            if noise_trace is not None:
                this_noise_trace = noise_trace

        fragment = this_trace[i0:i1].copy()
        if noise_trace is not None:
            noise_fragment = this_noise_trace[i0:i1].copy()

        raw_fragments.append(fragment)
        try:
            if noise_trace is None:
                filtered_fragments.append(func(fragment, **kwargs))
            else:
                filtered_fragments.append(func(fragment, noise_fragment, **kwargs))
                noise_fragments.append(noise_fragment)
        except ValueError:
            # Ignore if some fail
            pass

        used_edges.append(edges)

    return filtered_fragments, noise_fragments, raw_fragments, used_edges


def combine_trace_fragments(trace_fragments, window_edges_or_ind, full_size, window_func=scipy.signal.windows.hann):

    final_trace = np.zeros(full_size)
    num_overlaps = np.zeros(full_size)
    for fragment, edges_or_ind in zip(trace_fragments, window_edges_or_ind):
        if not len(edges_or_ind) == 2:
            raise NotImplementedError
        #     ind = np.arange(i0, i1)
        # else:
        #     ind = edges_or_ind

        i0, i1 = edges_or_ind
        if i1 > full_size:
            i1 = full_size
            fragment = fragment[:i1 - i0]

        weights = window_func(len(fragment))  # May be different on the last one
        final_trace[i0:i1] += weights * fragment
        num_overlaps[i0:i1] += weights

    final_trace /= num_overlaps
    return final_trace


def plot_psd(y, ax=None, label='', **kwargs):
    """
    Plot power spectral density

    Parameters
    ----------
    y
    ax
    label

    Returns
    -------

    """
    default_kwargs = dict(nperseg=256)
    default_kwargs.update(kwargs)
    fs = 1

    if ax is None:
        fig, ax = plt.subplots(dpi=100)
    f, Pxx_den = scipy.signal.welch(y, fs, **default_kwargs)
    ax.plot(f, Pxx_den, label=label)
    # ax2 = ax.twinx()
    # ax2.semilogy(f, Pxx_den, color='tab:orange')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    ax.legend()

    plt.title("PSD of full trace")

    return ax