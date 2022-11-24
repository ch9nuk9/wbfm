import numpy as np
from matplotlib import pyplot as plt

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column


def calc_triggered_average_indices(binary_state, min_duration, trace_len):
    if trace_len is not None:
        binary_state = binary_state[:trace_len]
    all_starts, all_ends = get_contiguous_blocks_from_column(binary_state, already_boolean=True)
    # Turn into time series
    all_ind = []
    for start, end in zip(all_starts, all_ends):
        if end - start <= min_duration:
            continue
        ind = np.arange(start, end)
        all_ind.append(ind)
    return all_ind


def calc_triggered_average_matrix(trace, all_ind):
    max_len_subset = max(map(len, all_ind))
    triggered_avg_matrix = np.zeros((len(all_ind), max_len_subset))
    triggered_avg_matrix[:] = np.nan
    for i, ind in enumerate(all_ind):
        triggered_avg_matrix[i, np.arange(len(ind))] = trace[ind]
    return triggered_avg_matrix


def calc_triggered_average_stats(triggered_avg_matrix):
    triggered_avg = np.nanmean(triggered_avg_matrix, axis=0)
    triggered_std = np.nanstd(triggered_avg_matrix, axis=0)
    triggered_avg_counts = np.nansum(~np.isnan(triggered_avg_matrix), axis=0)
    return triggered_avg, triggered_std, triggered_avg_counts


def plot_triggered_average_from_matrix_with_histogram(triggered_avg_matrix, show_individual_lines=True):
    triggered_avg, triggered_std, triggered_avg_counts = calc_triggered_average_stats(triggered_avg_matrix)

    fig, axes = plt.subplots(nrows=2, sharex=True, dpi=100)

    ax = axes[0]
    plot_triggered_average_from_matrix(triggered_avg_matrix, ax, show_individual_lines)

    triggered_avg_counts = np.nansum(~np.isnan(triggered_avg_matrix), axis=0)
    x = np.arange(len(triggered_avg))
    axes[1].bar(x, triggered_avg_counts)
    axes[1].set_ylabel("Num contributing")
    axes[1].set_xlabel("Time (frames)")

    return axes


def plot_triggered_average_from_matrix(triggered_avg_matrix, ax, show_individual_lines=True, **kwargs):
    triggered_avg = np.nanmean(triggered_avg_matrix, axis=0)
    triggered_std = np.nanstd(triggered_avg_matrix, axis=0)
    x = np.arange(len(triggered_avg))
    ax.plot(triggered_avg, **kwargs)
    if show_individual_lines:
        for trace in triggered_avg_matrix:
            ax.plot(trace, 'black', alpha=0.2)
    ax.fill_between(x, triggered_avg + triggered_std, triggered_avg - triggered_std, alpha=0.25)
    ax.set_ylabel("Activity")
