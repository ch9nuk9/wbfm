import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column


def calc_triggered_average_indices(binary_state, min_duration, trace_len, ind_preceding=0):
    """
    If ind_preceding > 0, then a very early event will lead to negative indices
    Thus in later steps, the trace should be padded with nan at the end to avoid wrapping

    Parameters
    ----------
    binary_state
    min_duration
    trace_len
    ind_preceding

    Returns
    -------

    """
    if trace_len is not None:
        binary_state = binary_state[:trace_len]
    all_starts, all_ends = get_contiguous_blocks_from_column(binary_state, already_boolean=True)
    # Turn into time series
    all_ind = []
    for start, end in zip(all_starts, all_ends):
        if end - start < min_duration:
            continue
        ind = np.arange(start - ind_preceding, end)
        all_ind.append(ind)
    return all_ind


def calc_triggered_average_matrix(trace, all_ind, mean_subtract=False):
    max_len_subset = max(map(len, all_ind))
    # Pad with nan in case there are negative indices, but only the end
    trace = np.pad(trace, max_len_subset, mode='constant', constant_values=(np.nan, np.nan))[max_len_subset:]
    triggered_avg_matrix = np.zeros((len(all_ind), max_len_subset))
    triggered_avg_matrix[:] = np.nan
    for i, ind in enumerate(all_ind):
        triggered_avg_matrix[i, np.arange(len(ind))] = trace[ind]
    if mean_subtract:
        triggered_avg_matrix -= np.nanmean(triggered_avg_matrix, axis=1, keepdims=True)
    return triggered_avg_matrix


def nan_points_of_state_before_point(triggered_average_mat, list_of_triggered_ind, preceding_ind,
                                     list_of_invalid_states,
                                     beh_annotations):
    """
    Checks points up to a certain level, and nans them if they are invalid. Only checks up to a certain threshold

    Parameters
    ----------
    mat
    preceding_ind
    list_of_invalid_states

    Returns
    -------

    """

    for i_trace in range(len(list_of_triggered_ind)):
        these_ind = list_of_triggered_ind[i_trace]
        for i_local, i_global in enumerate(these_ind):
            if i_local >= preceding_ind:
                break
            if beh_annotations[i_global] in list_of_invalid_states:
                triggered_average_mat[i_trace, i_local] = np.nan
                print("Removed points: ", i_trace, i_local)

    return triggered_average_mat


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


def plot_triggered_average_from_matrix(triggered_avg_matrix, ax, show_individual_lines=True, min_lines=2,
                                       **kwargs):
    triggered_avg, triggered_std, triggered_avg_counts = calc_triggered_average_stats(triggered_avg_matrix)
    # Remove points where there are too few lines contributing
    to_remove = triggered_avg_counts < min_lines
    triggered_avg[to_remove] = np.nan
    triggered_std[to_remove] = np.nan
    xmax = pd.Series(triggered_avg).last_valid_index()
    triggered_avg = triggered_avg[:xmax]
    triggered_std = triggered_std[:xmax]

    # Plot
    x = np.arange(xmax)
    ax.plot(triggered_avg, **kwargs)
    if show_individual_lines:
        for trace in triggered_avg_matrix:
            ax.plot(trace[:xmax], 'black', alpha=0.2)
    ax.fill_between(x, triggered_avg + triggered_std, triggered_avg - triggered_std, alpha=0.25)
    ax.set_ylabel("Activity")
    ax.set_ylim(np.nanmin(triggered_avg- triggered_std), np.nanmax(triggered_avg+ triggered_std))


def ax_plot_func_for_grid_plot(t, y, ax, name, project_data, state, min_lines=4, **kwargs):
    """
    Designed to be used with make_grid_plot_using_project with the arg ax_plot_func=ax_plot_func
    Note that you must create a closure to remove the following args, and pass a lambda:
        project_data
        state
        min_lines

    Example:
    func = lambda *args, **kwargs: ax_plot_func_for_grid_plot(*args, project_data=p, state=1, **kwargs)

    Parameters
    ----------
    project_data
    t
    y
    ax
    name
    kwargs

    Returns
    -------

    """
    ind_preceding = 20
    worm_class = project_data.worm_posture_class

    ind = worm_class.calc_triggered_average_indices(state=state, trace_len=len(y), ind_preceding=ind_preceding)
    mat = calc_triggered_average_matrix(y, ind)
    mat = nan_points_of_state_before_point(mat, ind, ind_preceding, [state],
                                           np.array(worm_class.behavior_annotations_fluorescence_fps))
    plot_triggered_average_from_matrix(mat, ax, show_individual_lines=True, label=name, min_lines=min_lines)
    ax.axhline(np.nanmean(mat), c='black', ls='--')
    ax.plot(ind_preceding, np.nanmean(mat), "r>", markersize=10)
