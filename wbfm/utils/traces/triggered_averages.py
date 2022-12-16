import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from backports.cached_property import cached_property
from matplotlib import pyplot as plt

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, remove_short_state_changes
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


@dataclass
class TriggeredAverageIndices:
    """
    Class for keeping track of all the settings related to a general triggered average

    Has all postprocessing functions, so that analysis is consistent when calculated for multiple traces

    The traces themselves are not stored here
    """
    # Initial calculation of indices
    behavioral_annotation: pd.Series
    behavioral_state: int
    min_duration: int
    ind_preceding: int

    max_duration: int = None
    gap_size_to_remove: int = None

    # Postprocessing the trace matrix (per trace)
    trace_len: int = None
    to_nan_points_of_state_before_point: bool = True
    min_lines: int = 2
    include_censored_data: bool = True  # To include events whose termination is after the end of the data

    DEBUG: bool = False

    @property
    def binary_state(self):
        binary_state = self.behavioral_annotation == self.behavioral_state
        if self.gap_size_to_remove is not None:
            binary_state = remove_short_state_changes(binary_state, self.gap_size_to_remove)
        return binary_state

    @property
    def triggered_average_indices(self):
        """
        If ind_preceding > 0, then a very early event will lead to negative indices
        Thus in later steps, the trace should be padded with nan at the end to avoid wrapping

        Parameters
        ----------

        Returns
        -------

        """
        if self.trace_len is not None:
            binary_state = self.binary_state[:self.trace_len]
        else:
            binary_state = self.binary_state
        all_starts, all_ends = get_contiguous_blocks_from_column(binary_state, already_boolean=True)
        # Turn into time series
        all_ind = []
        for start, end in zip(all_starts, all_ends):
            if self.DEBUG:
                print("Checking block: ", start, end)

            is_too_short = end - start < self.min_duration
            is_too_long = (self.max_duration is not None) and (end - start > self.max_duration)
            is_at_edge = start == 0
            starts_with_misannotation = self.behavioral_annotation.iat[start-1] == -1
            if is_too_short or is_too_long or is_at_edge or starts_with_misannotation:
                if self.DEBUG:
                    print("Skipping because: ", is_too_short, is_too_long, is_at_edge, starts_with_misannotation)
                continue
            ind = np.arange(start - self.ind_preceding, end)
            all_ind.append(ind)
        return all_ind

    def calc_triggered_average_matrix(self, trace, mean_subtract=False):
        all_ind = self.triggered_average_indices
        max_len_subset = max(map(len, all_ind))
        # Pad with nan in case there are negative indices, but only the end
        trace = np.pad(trace, max_len_subset, mode='constant', constant_values=(np.nan, np.nan))[max_len_subset:]
        triggered_avg_matrix = np.zeros((len(all_ind), max_len_subset))
        triggered_avg_matrix[:] = np.nan
        for i, ind in enumerate(all_ind):
            triggered_avg_matrix[i, np.arange(len(ind))] = trace[ind]
        if mean_subtract:
            triggered_avg_matrix -= np.nanmean(triggered_avg_matrix, axis=1, keepdims=True)
        if self.to_nan_points_of_state_before_point:
            triggered_avg_matrix = self.nan_points_of_state_before_point(triggered_avg_matrix)
        return triggered_avg_matrix

    def nan_points_of_state_before_point(self, triggered_average_mat):
        """
        Checks points up to a certain level, and nans them if they are invalid. Only checks up to a certain threshold

        Parameters
        ----------
        triggered_average_mat

        Returns
        -------

        """
        list_of_triggered_ind = self.triggered_average_indices
        list_of_invalid_states = [self.behavioral_state, -1]
        beh_annotations = self.behavioral_annotation.values
        for i_trace in range(len(list_of_triggered_ind)):
            these_ind = list_of_triggered_ind[i_trace]
            for i_local, i_global in enumerate(these_ind):
                if i_global < 0:
                    continue
                if i_local >= self.ind_preceding:
                    break
                if beh_annotations[i_global] in list_of_invalid_states:
                    # Remove all points before this
                    for i_to_remove in range(i_local):
                        triggered_average_mat[i_trace, i_to_remove] = np.nan
        return triggered_average_mat

    def prep_triggered_average_for_plotting(self, triggered_avg_matrix):
        triggered_avg, triggered_std, triggered_avg_counts = self.calc_triggered_average_stats(triggered_avg_matrix)
        # Remove points where there are too few lines contributing
        to_remove = triggered_avg_counts < self.min_lines
        triggered_avg[to_remove] = np.nan
        triggered_std[to_remove] = np.nan
        xmax = pd.Series(triggered_avg).last_valid_index()
        triggered_avg = triggered_avg[:xmax]
        raw_trace_mean = np.nanmean(triggered_avg)
        # triggered_avg -= raw_trace_mean  # No y axis is shown, so this is only for later calculation cleanup
        triggered_std = triggered_std[:xmax]
        is_valid = len(triggered_avg) > 0 and np.count_nonzero(~np.isnan(triggered_avg)) > 0
        return raw_trace_mean, triggered_avg, triggered_std, xmax, is_valid

    @staticmethod
    def calc_triggered_average_stats(triggered_avg_matrix):
        triggered_avg = np.nanmean(triggered_avg_matrix, axis=0)
        triggered_std = np.nanstd(triggered_avg_matrix, axis=0)
        triggered_avg_counts = np.nansum(~np.isnan(triggered_avg_matrix), axis=0)
        return triggered_avg, triggered_std, triggered_avg_counts

    def calc_significant_points_from_triggered_matrix(self, triggered_avg_matrix):
        """
        Calculates the time points that are (based on the std) "significantly" different from a flat line

        Designed to be used to remove uninteresting traces from triggered average grid plots

        Parameters
        ----------
        triggered_avg_matrix
        min_lines

        Returns
        -------

        """
        raw_trace_mean, triggered_avg, triggered_std, xmax, is_valid = \
            self.prep_triggered_average_for_plotting(triggered_avg_matrix)
        if not is_valid:
            return []
        upper_shading = triggered_avg + triggered_std
        lower_shading = triggered_avg - triggered_std
        x_significant = np.where(np.logical_or(lower_shading > raw_trace_mean, upper_shading < raw_trace_mean))[0]
        return x_significant

    def plot_triggered_average_from_matrix(self, triggered_avg_matrix, ax=None,
                                           show_individual_lines=True,
                                           color_significant_times=False,
                                           **kwargs):
        raw_trace_mean, triggered_avg, triggered_std, xmax, is_valid = \
            self.prep_triggered_average_for_plotting(triggered_avg_matrix)
        if not is_valid:
            logging.warning("Found invalid neuron (empty triggered average)")
            return

        # Plot
        if ax is None:
            fig, ax = plt.subplots(dpi=100)
        x = np.arange(xmax)
        # Lines
        ax.plot(triggered_avg, **kwargs)
        if show_individual_lines:
            for trace in triggered_avg_matrix:
                ax.plot(trace[:xmax], 'black', alpha=0.2)
        # Shading
        upper_shading = triggered_avg + triggered_std
        lower_shading = triggered_avg - triggered_std
        ax.fill_between(x, upper_shading, lower_shading, alpha=0.25)
        ax.set_ylabel("Activity")
        ax.set_ylim(np.nanmin(lower_shading), np.nanmax(upper_shading))
        # Reference points
        ax.axhline(raw_trace_mean, c='black', ls='--')
        ax.axvline(x=self.ind_preceding, color='r', ls='--')
        # ax.plot(self.ind_preceding, raw_trace_mean, "r>", markersize=10)
        # Optional orange points
        x_significant = self.calc_significant_points_from_triggered_matrix(triggered_avg_matrix)
        if color_significant_times:
            if len(x_significant) > 0:
                ax.plot(x_significant, triggered_avg[x_significant], 'o', color='tab:orange')

    def onset_vector(self):
        local_idx_of_onset = self.ind_preceding
        idx_onsets = np.array([vec[local_idx_of_onset] for vec in self.triggered_average_indices if
                               vec[local_idx_of_onset] > 0])
        onset_vec = np.zeros(self.trace_len)
        onset_vec[idx_onsets] = 1
        return onset_vec


@dataclass
class FullDatasetTriggeredAverages:
    """
    A class that uses TriggeredAverageIndices to process each trace of a full dataset (Dataframe) into a matrix of
    triggered averages

    Also has functions for plotting
    """
    df_traces: pd.DataFrame

    # Calculating indices
    ind_class: TriggeredAverageIndices

    # Calculating full average
    mean_subtract_each_trace: bool = False
    min_lines: int = 2
    min_points_for_significance: int = 5

    # Plotting
    show_individual_lines: bool = True
    color_significant_times: bool = True

    @property
    def neuron_names(self):
        return get_names_from_df(self.df_traces)

    def triggered_average_matrix(self, name):
        return self.ind_class.calc_triggered_average_matrix(self.df_traces[name])

    def which_neurons_are_significant(self, min_points_for_significance=None):
        if min_points_for_significance is not None:
            self.min_points_for_significance = min_points_for_significance
        names_to_keep = []
        for name in self.neuron_names:
            mat = self.triggered_average_matrix(name)
            x_significant = self.ind_class.calc_significant_points_from_triggered_matrix(mat)
            if len(x_significant) > self.min_points_for_significance:
                names_to_keep.append(name)

        if len(names_to_keep) == 0:
            logging.warning("Found no significant neurons, subsequent steps may not work")

        return names_to_keep

    def ax_plot_func_for_grid_plot(self, t, y, ax, name, **kwargs):
        """Same as ax_plot_func_for_grid_plot, but can be used directly"""
        plot_kwargs = dict(show_individual_lines=True, color_significant_times=True, label=name)
        plot_kwargs.update(kwargs)

        mat = self.ind_class.calc_triggered_average_matrix(y)
        self.ind_class.plot_triggered_average_from_matrix(mat, ax, **plot_kwargs)
        ax.axhline(0, c='black', ls='--')
        ax.plot(self.ind_class.ind_preceding, 0, "r>", markersize=10)


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
    state: the state whose onset is calculated as the trigger
    min_lines: the minimum number of lines that must exist for a line to be plotted
    project_data
    t: time vector (unused)
    y: full trace (1d)
    ax: matplotlib axis
    name: neuron name
    kwargs

    Returns
    -------

    """
    plot_kwargs = dict(show_individual_lines=True, color_significant_times=True, label=name)
    plot_kwargs.update(kwargs)

    ind_preceding = 20
    worm_class = project_data.worm_posture_class

    ind_class = worm_class.calc_triggered_average_indices(state=state, ind_preceding=ind_preceding,
                                                          min_lines=min_lines)
    mat = ind_class.calc_triggered_average_matrix(y)
    ind_class.plot_triggered_average_from_matrix(mat, ax, **plot_kwargs)


# def plot_triggered_average_from_matrix_with_histogram(triggered_avg_matrix, show_individual_lines=True):
#     triggered_avg, triggered_std, triggered_avg_counts = calc_triggered_average_stats(triggered_avg_matrix)
#
#     fig, axes = plt.subplots(nrows=2, sharex=True, dpi=100)
#
#     ax = axes[0]
#     plot_triggered_average_from_matrix(triggered_avg_matrix, ax, show_individual_lines)
#
#     triggered_avg_counts = np.nansum(~np.isnan(triggered_avg_matrix), axis=0)
#     x = np.arange(len(triggered_avg))
#     axes[1].bar(x, triggered_avg_counts)
#     axes[1].set_ylabel("Num contributing")
#     axes[1].set_xlabel("Time (frames)")
#
#     return axes


def assign_id_based_on_closest_onset_in_split_lists(short_onsets, long_onsets, rev_onsets) -> dict:
    """
    Assigns each reversal a class based on which list contains an event closes to that reversal

    Note if a reversal has no previous forward, it will be removed!

    Parameters
    ----------
    short_onsets
    long_onsets
    rev_onsets

    Returns
    -------

    """
    dict_of_rev_with_id = {}
    for rev in rev_onsets:
        # For both forward lists, get the previous indices via subtraction
        these_short = short_onsets.copy() - rev
        these_short = these_short[these_short < 0]

        these_long = long_onsets.copy() - rev
        these_long = these_long[these_long < 0]

        # Then the smaller absolute one (closer in time) one gives the class
        no_prev_short = len(these_short) == 0
        no_prev_long = len(these_long) == 0
        if (no_prev_long and not no_prev_short) or np.min(np.abs(these_short)) < np.min(np.abs(these_long)):
            dict_of_rev_with_id[rev] = 0
        else:
            dict_of_rev_with_id[rev] = 1

        # Optimization: Finally, remove the used one from the fwd onset list

    return dict_of_rev_with_id
