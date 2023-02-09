import logging
import time
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import scipy
from backports.cached_property import cached_property
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from wbfm.utils.external.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, remove_short_state_changes
from wbfm.utils.external.utils_zeta_statistics import calculate_zeta_cumsum, jitter_indices, calculate_p_value_from_zeta
from wbfm.utils.general.utils_matplotlib import paired_boxplot_from_dataframes


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
    dict_of_events_to_keep: dict = None

    DEBUG: bool = False

    @property
    def binary_state(self):
        binary_state = self.behavioral_annotation == self.behavioral_state
        if self.gap_size_to_remove is not None:
            binary_state = remove_short_state_changes(binary_state, self.gap_size_to_remove)
        return binary_state

    @cached_property
    def cleaned_binary_state(self):
        if self.trace_len is not None:
            return self.binary_state.iloc[:self.trace_len]
        else:
            return self.binary_state

    def triggered_average_indices(self, dict_of_events_to_keep=None, DEBUG=False) -> list:
        """
        Calculates triggered average indices based on a binary state vector saved in this class

        If ind_preceding > 0, then a very early event will lead to negative indices
        Thus in later steps, the trace should be padded with nan at the end to avoid wrapping

        Parameters
        ----------
        dict_of_events_to_keep: Optional dict determining a subset of indices to keep. Key=state starts, value=0 or 1
            Example:
            all_starts = [15, 66, 114, 130]
            dict_of_ind_to_keep = {15: 0, 66: 1, 114: 0}

            Note that not all starts need to be in dict_of_ind_to_keep; missing entries are dropped by default

        Returns
        -------

        """
        if dict_of_events_to_keep is None:
            dict_of_events_to_keep = self.dict_of_events_to_keep
        else:
            self.dict_of_events_to_keep = dict_of_events_to_keep
        binary_state = self.cleaned_binary_state.copy()
        all_starts, all_ends = get_contiguous_blocks_from_column(binary_state,
                                                                 already_boolean=True, skip_boolean_check=True)
        # Turn into time series
        all_ind = []
        beh_vec = self.behavioral_annotation.to_numpy()
        for start, end in zip(all_starts, all_ends):
            if DEBUG:
                print("Checking block: ", start, end)
            is_too_short = end - start < self.min_duration
            is_too_long = (self.max_duration is not None) and (end - start > self.max_duration)
            is_at_edge = start == 0
            starts_with_misannotation = beh_vec[start-1] == BehaviorCodes.UNKNOWN
            not_in_dict = (dict_of_events_to_keep is not None) and (dict_of_events_to_keep.get(start, 0) == 0)
            if is_too_short or is_too_long or is_at_edge or starts_with_misannotation or not_in_dict:
                if DEBUG:
                    print("Skipping because: ", is_too_short, is_too_long, is_at_edge, starts_with_misannotation)
                continue
            elif DEBUG:
                print("***Keeping***")
            ind = np.arange(start - self.ind_preceding, end)
            all_ind.append(ind)
        return all_ind

    def calc_triggered_average_matrix(self, trace, mean_subtract=False, custom_ind: List[np.ndarray]=None,
                                      nan_times_with_too_few=False, max_len=None,
                                      **ind_kwargs):
        """
        Uses triggered_average_indices to extract a matrix of traces at each index, with nan padding to equalize the
        lengths of the traces


        Parameters
        ----------
        trace
        mean_subtract
        custom_ind: instead of using self.triggered_average_indices. If not None, ind_kwargs are not used
        nan_times_with_too_few
        max_len: Cut off matrix at a time point. Usually if there aren't enough data points that far
        ind_kwargs

        Returns
        -------

        """
        if custom_ind is None:
            all_ind = self.triggered_average_indices(**ind_kwargs)
        else:
            all_ind = custom_ind
        if max_len is None:
            max_len_subset = max(map(len, all_ind))
        else:
            max_len_subset = max_len
        # Pad with nan in case there are negative indices, but only the end
        trace = np.pad(trace, max_len_subset, mode='constant', constant_values=(np.nan, np.nan))[max_len_subset:]
        triggered_avg_matrix = np.zeros((len(all_ind), max_len_subset))
        triggered_avg_matrix[:] = np.nan
        # Save either entire traces, or traces up to a point
        for i, ind in enumerate(all_ind):
            if max_len is not None:
                ind = ind.copy()[:max_len]
            triggered_avg_matrix[i, np.arange(len(ind))] = trace[ind]

        # Postprocessing
        if mean_subtract:
            triggered_avg_matrix -= np.nanmean(triggered_avg_matrix, axis=1, keepdims=True)
        if self.to_nan_points_of_state_before_point:
            triggered_avg_matrix = self.nan_points_of_state_before_point(triggered_avg_matrix, all_ind)
        if nan_times_with_too_few:
            num_lines_at_each_time = np.sum(~np.isnan(triggered_avg_matrix), axis=0)
            times_to_remove = num_lines_at_each_time < self.min_lines
            triggered_avg_matrix[:, times_to_remove] = np.nan

        return triggered_avg_matrix

    def calc_null_triggered_average_matrix(self, trace, **kwargs):
        """Similar to calc_triggered_average_matrix, but jitters the indices"""
        triggered_average_indices = self.triggered_average_indices()
        ind_jitter = jitter_indices(triggered_average_indices, max_jitter=len(trace), max_len=len(trace))
        mat_jitter = self.calc_triggered_average_matrix(trace, custom_ind=ind_jitter, **kwargs)
        return mat_jitter

    def nan_points_of_state_before_point(self, triggered_average_mat, list_of_triggered_ind):
        """
        Checks points up to a certain level, and nans them if they are invalid. Only checks up to a certain threshold

        Parameters
        ----------
        triggered_average_mat

        Returns
        -------

        """
        invalid_states = {self.behavioral_state, BehaviorCodes.UNKNOWN}
        beh_annotations = self.behavioral_annotation.to_numpy()
        for i_trace in range(len(list_of_triggered_ind)):
            these_ind = list_of_triggered_ind[i_trace]
            for i_local, i_global in enumerate(these_ind):
                if i_global < 0:
                    continue
                if i_local >= self.ind_preceding:
                    break
                if beh_annotations[i_global] in invalid_states:
                    # Remove all points before this
                    for i_to_remove in range(i_local + 1):
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

    def calc_p_value_using_zeta(self, trace, num_baseline_lines=100, DEBUG=False) -> float:
        """
        See utils_zeta_statistics. Following:
        https://elifesciences.org/articles/71969#

        Parameters
        ----------
        trace
        num_baseline_lines

        Returns
        -------

        """
        # Original triggered average matrix
        triggered_average_indices = self.triggered_average_indices()
        # Set max number of time points based on number of lines present
        # In other words, find the max point in time when there are still enough lines
        if self.min_lines > 0:
            all_lens = np.array(list(map(len, triggered_average_indices)))
            ind_lens_enough = np.argsort(all_lens)[:-self.min_lines]
            max_matrix_length = np.max(all_lens[ind_lens_enough])
        else:
            max_matrix_length = None
        mat = self.calc_triggered_average_matrix(trace, custom_ind=triggered_average_indices,
                                                 max_len=max_matrix_length)
        zeta_line_dat = calculate_zeta_cumsum(mat, DEBUG=DEBUG)

        if DEBUG:
            print(max_matrix_length)

            plt.figure(dpi=100)
            self.plot_triggered_average_from_matrix(mat, show_individual_lines=True)
            plt.title("Triggered average")

            plt.figure(dpi=100)
            plt.plot(np.sum(~np.isnan(mat), axis=0))
            plt.title("Number of lines contributing to each point")
            plt.show()

        # Null distribution
        if max_matrix_length is None:
            mat_len = mat.shape[1]
        else:
            mat_len = max_matrix_length
        baseline_lines = self.calc_null_distribution_of_triggered_lines(mat_len,
                                                                        num_baseline_lines, trace,
                                                                        triggered_average_indices)

        # if DEBUG:
        #     plt.figure(dpi=100)
        #     all_ind_jitter = np.hstack(all_ind_jitter)
        #     plt.hist(all_ind_jitter)
        #     plt.title("Number of times each data point is selected")
        #     plt.show()

        # Normalize by the std of the baseline
        # Note: calc the std across trials, then average across time
        baseline_per_line_std = np.std(baseline_lines, axis=0)
        baseline_std = np.mean(baseline_per_line_std)

        zeta_line_dat /= baseline_std
        baseline_lines /= baseline_std

        if DEBUG:
            plt.figure(dpi=100)
            plt.plot(zeta_line_dat)
            for i_row in range(baseline_lines.shape[0]):
                line = baseline_lines[i_row, :]
                plt.plot(line, 'gray', alpha=0.1)
            plt.ylabel("Deviation (std of baseline)")
            plt.title("Trace zeta line and null distribution")
            plt.show()

        # Calculate individual zeta values (max deviation)
        zeta_dat = np.max(np.abs(zeta_line_dat))
        zetas_baseline = np.max(np.abs(baseline_lines), axis=1)

        # ALT: calculate sum of squares, and plot
        # Idea: maybe I can do chi squared instead
        # Following: https://stats.stackexchange.com/questions/200886/what-is-the-distribution-of-sum-of-squared-errors
        # if DEBUG:
        #     zeta2_dat = np.sum(np.abs(zeta_line_dat)**2.0)
        #     zetas2_baseline = np.sum(np.abs(baseline_lines)**2.0, axis=1)
        #
        #     # What is the df for time series errors?
        #     p2 = 1 - scipy.stats.chi2.cdf(zeta2_dat, 2)
        #
        #     plt.figure(dpi=100)
        #     plt.hist(zetas2_baseline)#, bins=np.arange(0, np.max(zetas2_baseline)))
        #     plt.vlines(zeta2_dat, 0, len(zetas_baseline) / 2, colors='red')
        #     plt.title(f"Distribution of sum of squares, with p={p2}")
        #     plt.show()

        # Final p value
        p = calculate_p_value_from_zeta(zeta_dat, zetas_baseline)

        if DEBUG:
            plt.figure(dpi=100)
            plt.hist(zetas_baseline)
            plt.vlines(zeta_dat, 0, len(zetas_baseline) / 2, colors='red')
            plt.title(f"Distribution of maxima of null, with p value: {p}")
            plt.show()

        return p

    def calc_null_distribution_of_triggered_lines(self, max_matrix_length, num_baseline_lines, trace,
                                                  triggered_average_indices):
        baseline_lines = np.zeros((num_baseline_lines, max_matrix_length))
        all_ind_jitter = []
        for i in range(num_baseline_lines):
            ind_jitter = jitter_indices(triggered_average_indices, max_jitter=len(trace), max_len=len(trace))
            mat_jitter = self.calc_triggered_average_matrix(trace, custom_ind=ind_jitter,
                                                            max_len=max_matrix_length)
            zeta_line = calculate_zeta_cumsum(mat_jitter)
            baseline_lines[i, :] = zeta_line
            all_ind_jitter.extend(ind_jitter)
            # if DEBUG:
            #     time.sleep(2)
        return baseline_lines

    def calc_p_value_using_ttest(self, trace, gap=5, DEBUG=False) -> float:
        """
        Calculates a p value using a paired t-test on the pre- and post-stimulus time periods

        Note that this is generally sensitive to ind_preceding (in addition to other arguments0

        Parameters
        ----------
        trace
        num_baseline_lines

        Returns
        -------

        """
        mat = self.calc_triggered_average_matrix(trace)
        means_before, means_after = self.split_means_from_triggered_average_matrix(mat, gap=gap)
        p = scipy.stats.ttest_rel(means_before, means_after, nan_policy='omit').pvalue

        if DEBUG:
            self.plot_triggered_average_from_matrix(mat, show_individual_lines=True)
            plt.title(f"P value: {p}")

            df = pd.DataFrame([means_before, means_after]).dropna(axis=1)
            paired_boxplot_from_dataframes(df)
            plt.title(f"P value: {p}")

            plt.show()

        return p

    def split_means_from_triggered_average_matrix(self, mat, gap):
        """Gets mean of trace before and after the trigger (same window length)"""
        i_trigger = self.ind_preceding
        num_pts = i_trigger - gap
        means_before = np.nanmean(mat[:, 0:num_pts], axis=1)
        means_after = np.nanmean(mat[:, i_trigger:i_trigger + num_pts], axis=1)
        return means_before, means_after

    def plot_triggered_average_from_matrix(self, triggered_avg_matrix, ax=None,
                                           show_individual_lines=False,
                                           color_significant_times=False,
                                           is_second_plot=False,
                                           **kwargs):
        """
        Core plotting function; must be passed a matrix

        Parameters
        ----------
        triggered_avg_matrix
        ax
        show_individual_lines
        color_significant_times
        kwargs

        Returns
        -------

        """
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

        if not is_second_plot:
            ax.set_ylabel("Activity")
            ax.set_ylim(np.nanmin(lower_shading), np.nanmax(upper_shading))
            # Reference points
            ax.axhline(raw_trace_mean, c='black', ls='--')
            ax.axvline(x=self.ind_preceding, color='r', ls='--')
        else:
            ax.autoscale()
        # Optional orange points
        x_significant = self.calc_significant_points_from_triggered_matrix(triggered_avg_matrix)
        if color_significant_times:
            if len(x_significant) > 0:
                ax.plot(x_significant, triggered_avg[x_significant], 'o', color='tab:orange')

    def plot_ind_over_trace(self, trace):
        """
        Plots the indices stored here over a trace (for debugging)

        Parameters
        ----------
        trace

        Returns
        -------

        """

        plt.figure(dpi=100)
        plt.plot(trace)

        for ind in self.triggered_average_indices():
            ind = np.array(ind)
            ind = ind[ind > 0]
            plt.plot(ind, trace[ind], 'o', color='tab:orange')

    @property
    def idx_onsets(self):
        local_idx_of_onset = self.ind_preceding
        idx_onsets = np.array([vec[local_idx_of_onset] for vec in self.triggered_average_indices() if
                               vec[local_idx_of_onset] > 0])
        return idx_onsets

    def onset_vector(self):
        onset_vec = np.zeros(self.trace_len)
        onset_vec[self.idx_onsets] = 1
        return onset_vec

    @property
    def num_events(self):
        return len(self.idx_onsets)


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
    significance_calculation_method: str = 'zeta'  # Or: 'num_points'

    # Plotting
    show_individual_lines: bool = True
    color_significant_times: bool = True

    @property
    def neuron_names(self):
        names = list(set(self.df_traces.columns.get_level_values(0)))
        names.sort()
        return names

    def triggered_average_matrix(self, name):
        return self.ind_class.calc_triggered_average_matrix(self.df_traces[name])

    def which_neurons_are_significant(self, min_points_for_significance=None, num_baseline_lines=100,
                                      ttest_gap=5):
        if min_points_for_significance is not None:
            self.min_points_for_significance = min_points_for_significance
        names_to_keep = []
        all_p_values = {}
        all_effect_sizes = {}
        for name in tqdm(self.neuron_names, leave=False):

            if self.significance_calculation_method == 'zeta':
                logging.warning("Zeta calculation is unstable for calcium imaging!")
                trace = self.df_traces[name]
                p = self.ind_class.calc_p_value_using_zeta(trace, num_baseline_lines)
                all_p_values[name] = p
                to_keep = p < 0.05
            elif self.significance_calculation_method == 'num_points':
                logging.warning("Number of points calculation is not statistically justified!")
                mat = self.triggered_average_matrix(name)
                x_significant = self.ind_class.calc_significant_points_from_triggered_matrix(mat)
                all_p_values[name] = x_significant
                to_keep = len(x_significant) > self.min_points_for_significance
            elif self.significance_calculation_method == 'ttest':
                trace = self.df_traces[name]
                p = self.ind_class.calc_p_value_using_ttest(trace, ttest_gap)
                all_p_values[name] = p
                to_keep = p < 0.05
            else:
                raise NotImplementedError(f"Unrecognized significance_calculation_method: "
                                          f"{self.significance_calculation_method}")

            if to_keep:
                names_to_keep.append(name)

        if len(names_to_keep) == 0:
            logging.warning("Found no significant neurons, subsequent steps may not work")

        return names_to_keep, all_p_values, all_effect_sizes

    def ax_plot_func_for_grid_plot(self, t, y, ax, name, **kwargs):
        """Same as ax_plot_func_for_grid_plot, but can be used directly"""
        if kwargs.get('is_second_plot', False):
            # Do not want two legend labels
            if 'label' in kwargs:
                kwargs['label'] = ''
            plot_kwargs = dict(label='')
        else:
            plot_kwargs = dict(label=name)
        plot_kwargs.update(kwargs)

        mat = self.ind_class.calc_triggered_average_matrix(y)
        self.ind_class.plot_triggered_average_from_matrix(mat, ax, **plot_kwargs)
        # ax.axhline(0, c='black', ls='--')
        # ax.plot(self.ind_class.ind_preceding, 0, "r>", markersize=10)

    @staticmethod
    def load_from_project(project_data, trigger_opt=None, trace_opt=None, **kwargs):
        if trigger_opt is None:
            trigger_opt = {}
        if trace_opt is None:
            trace_opt = {}

        trigger_opt_default = dict(min_lines=3, ind_preceding=20)
        trigger_opt_default.update(trigger_opt)
        ind_class = project_data.worm_posture_class.calc_triggered_average_indices(**trigger_opt_default)

        trace_opt_default = dict(channel_mode='dr_over_r_20', calculation_mode='integration', min_nonnan=0.9)
        trace_opt_default.update(trace_opt)
        df_traces = project_data.calc_default_traces(**trace_opt_default)

        triggered_averages_class = FullDatasetTriggeredAverages(df_traces, ind_class, **kwargs)

        return triggered_averages_class


def ax_plot_func_for_grid_plot(t, y, ax, name, project_data, state, min_lines=4, **kwargs):
    """
    Designed to be used with make_grid_plot_using_project with the arg ax_plot_func=ax_plot_func
    Note that you must create a closure to remove the following args, and pass a lambda:
        project_data
        state
        min_lines

    Example:
    from functools import partial
    func = partial(ax_plot_func_for_grid_plot, project_data=p, state=1)

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
    plot_kwargs = dict(label=name)
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


def assign_id_based_on_closest_onset_in_split_lists(class1_onsets, class0_onsets, rev_onsets) -> dict:
    """
    Assigns each reversal a class based on which list contains an event closes to that reversal

    Note if a reversal has no previous forward, it will be removed!

    Parameters
    ----------
    class1_onsets
    class0_onsets
    rev_onsets

    Returns
    -------

    """
    raise ValueError("Not working! See test")
    dict_of_rev_with_id = {}
    for rev in rev_onsets:
        # For both forward lists, get the previous indices
        these_class0 = class0_onsets.copy() - rev
        these_class0 = these_class0[these_class0 < 0]

        these_class1 = class1_onsets.copy() - rev
        these_class1 = these_class1[these_class1 < 0]

        # Then the smaller absolute one (closer in time) one gives the class
        only_prev_short = len(these_class0) == 0 and len(these_class1) > 0
        only_prev_long = len(these_class1) == 0 and len(these_class0) > 0
        # Do not immediately calculate, because the list may be empty
        short_is_closer = lambda: np.min(np.abs(these_class0)) < np.min(np.abs(these_class1))
        if only_prev_short:
            dict_of_rev_with_id[rev] = 0
        elif only_prev_long:
            dict_of_rev_with_id[rev] = 1
        elif short_is_closer():
            # Need to check the above two conditions before trying to evaluate this
            dict_of_rev_with_id[rev] = 0
        else:
            dict_of_rev_with_id[rev] = 1

        # Optimization: Finally, remove the used one from the fwd onset list

    return dict_of_rev_with_id
