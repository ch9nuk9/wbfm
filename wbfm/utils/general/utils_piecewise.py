import numpy as np
import scipy
from matplotlib import pyplot as plt


def rolling_filter_trace_using_func(y, x, func, window, delta, **kwargs):
    """

    Parameters
    ----------
    y - trace (target)
    x - noise or predictor
    func - function for filtering. Takes y and x and kwargs
    window - size of rolling window
    kwargs

    Returns
    -------

    """
    edges = build_window_edges(len(y), window=window, delta=delta)
    filtered_fragments, noise_fragments, raw_fragments = apply_function_to_windows(y, edges, func, noise_trace=x,
                                                                                   **kwargs)
    y_filt = combine_trace_fragments(filtered_fragments, edges, len(y))

    return y_filt


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
    for edges in window_edges:
        i0, i1 = edges
        if i1 > len(trace):
            pad_len = i1 - len(trace)
            num_real_pts = i1 - i0 - pad_len
            if num_real_pts < min_pts_required:
                print("Skipping window; too few points")
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
        if noise_trace is None:
            filtered_fragments.append(func(fragment, **kwargs))
        else:
            filtered_fragments.append(func(fragment, noise_fragment, **kwargs))
            noise_fragments.append(noise_fragment)

    return filtered_fragments, noise_fragments, raw_fragments


def combine_trace_fragments(trace_fragments, window_edges, full_size, window_func=scipy.signal.windows.hann):
    final_trace = np.zeros(full_size)
    num_overlaps = np.zeros(full_size)
    for fragment, edges in zip(trace_fragments, window_edges):
        i0, i1 = edges
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