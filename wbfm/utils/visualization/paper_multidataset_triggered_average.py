"""
Designed to plot the triggered average of the paper's datasets.
"""
import itertools
import os
from collections import defaultdict
from dataclasses import dataclass, field
import random
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import split_flattened_index
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, shade_triggered_average
from wbfm.utils.general.utils_paper import apply_figure_settings
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.traces.triggered_averages import clustered_triggered_averages_from_list_of_projects, \
    ClusteredTriggeredAverages, plot_triggered_average_from_matrix_low_level
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


@dataclass
class PaperColoredTracePlotter:
    """
    Class to plot the colored traces of the paper's datasets.

    Specifically for raw/global/residual decompositions
    """

    def get_color_from_trigger_type(self, trigger_type):
        cmap = plt.get_cmap('tab10')
        color_mapping = {'raw_rev': cmap(0),
                         'raw': cmap(0),
                         'raw_fwd': cmap(0),
                         'raw_vt': cmap(0),
                         'raw_dt': cmap(0),
                         'global_rev': cmap(3),
                         'global': cmap(3),
                         'global_fwd': cmap(3),
                         'residual': cmap(4),
                         'residual_collision': cmap(4),
                         'residual_rectified_fwd': cmap(4),
                         'residual_rectified_rev': cmap(4),
                         'kymo': 'black',
                         'stimulus': 'black'}
        if trigger_type not in color_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(color_mapping.keys())}')
        return color_mapping[trigger_type]

    def get_trace_opt(self, **kwargs):
        trace_opt = dict(interpolate_nan=True, channel_mode='dr_over_r_50', remove_outliers=True,
                         rename_neurons_using_manual_ids=True, manual_id_confidence_threshold=0,
                         min_nonnan=0.8)
        trace_opt.update(kwargs)
        return trace_opt

    @classmethod
    def get_behavior_color_from_neuron_name(cls, neuron_name):
        """
        Returns the color of the cluster based on the neuron name.

        Parameters
        ----------
        neuron_name

        Returns
        -------

        """
        color_mapping = dict(
            RIS='tab:blue',
            AVA='tab:orange',
            RIV='tab:green',
            RMEL='tab:green',
            RMER='tab:green',
            SMDV='tab:green',
            RID='tab:red',
            RME='tab:purple',
            VB01='tab:purple',
            VB02='tab:purple',
            VB03='tab:purple',
            DB01='tab:purple',
            DB02='tab:purple',
            AVB='tab:red',
            RIB='tab:red',
            IL1L='black',
            IL2L='black',
        )
        # Add keys by adding the L/R and V/D suffixes
        for k in list(color_mapping.keys()):
            for suffix in ['L', 'R', 'V', 'D']:
                # Only add if not already in the dictionary
                if k + suffix not in color_mapping:
                    color_mapping[k + suffix] = color_mapping[k]
        if neuron_name not in color_mapping:
            raise ValueError(f"Neuron name {neuron_name} not found in color mapping")
        return color_mapping[neuron_name]


@dataclass
class PaperMultiDatasetTriggeredAverage(PaperColoredTracePlotter):
    """
    Class to plot the triggered average of the paper's datasets.

    Specifically designed for residual figures, and uses the proper colors for each type of triggered average.
    """

    all_projects: Dict[str, ProjectData]

    # Options for traces
    calculate_residual: bool = True
    min_nonnan: Optional[float] = 0.8

    # Three different sets of parameters: raw, global, and residual
    dataset_clusterer_dict: Dict[str, ClusteredTriggeredAverages] = None
    intermediates_dict: Dict[str, tuple] = None

    trace_opt: Optional[dict] = None
    trigger_opt: dict = field(default_factory=dict)

    # Optional: stimulus
    calc_stimulus: bool = False

    def __post_init__(self):
        # Analyze the project data to get the clusterers and intermediates
        trace_base_opt = self.get_trace_opt(min_nonnan=self.min_nonnan)
        trace_base_opt['use_paper_options'] = True
        if self.trace_opt is not None:
            trace_base_opt.update(self.trace_opt)

        self.dataset_clusterer_dict = defaultdict(None)
        self.intermediates_dict = defaultdict(lambda: (None, None, None))

        if self.calculate_residual:
            try:
                # Note: these won't work for immobilized data

                # Fast (residual)
                trigger_opt = dict(use_hilbert_phase=True, state=None)
                trigger_opt.update(self.trigger_opt)
                trace_opt = dict(residual_mode='pca')
                trace_opt.update(trace_base_opt)
                out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                         trace_opt=trace_opt)
                self.dataset_clusterer_dict['residual'] = out[0]
                self.intermediates_dict['residual'] = out[1]

                # Residual rectified forward
                trigger_opt['only_allow_events_during_state'] = BehaviorCodes.FWD
                cluster_opt = {}
                out = clustered_triggered_averages_from_list_of_projects(self.all_projects, cluster_opt=cluster_opt,
                                                                         trigger_opt=trigger_opt, trace_opt=trace_opt)
                self.dataset_clusterer_dict['residual_rectified_fwd'] = out[0]
                self.intermediates_dict['residual_rectified_fwd'] = out[1]

                # Residual rectified reverse
                trigger_opt['only_allow_events_during_state'] = BehaviorCodes.REV
                cluster_opt = {}
                out = clustered_triggered_averages_from_list_of_projects(self.all_projects, cluster_opt=cluster_opt,
                                                                         trigger_opt=trigger_opt, trace_opt=trace_opt)
                self.dataset_clusterer_dict['residual_rectified_rev'] = out[0]
                self.intermediates_dict['residual_rectified_rev'] = out[1]

                # Only used for BAG: self-collision triggered
                trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.SELF_COLLISION)
                trigger_opt.update(self.trigger_opt)
                trace_opt = dict(residual_mode='pca')
                trace_opt.update(trace_base_opt)
                out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                         trace_opt=trace_opt)
                self.dataset_clusterer_dict['residual_collision'] = out[0]
                self.intermediates_dict['residual_collision'] = out[1]

            except TypeError:
                print("Hilbert triggered averages failed; this may be because the data is immobilized")
                print("Only 'global' triggered averages will be available")

        # Slow reversal triggered (global)
        trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.REV)
        trigger_opt.update(self.trigger_opt)
        trace_opt = dict(residual_mode='pca_global')
        trace_opt.update(trace_base_opt)
        out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                 trace_opt=trace_opt)
        self.dataset_clusterer_dict['global_rev'] = out[0]
        self.intermediates_dict['global_rev'] = out[1]

        # Slow forward triggered (global)
        trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.FWD)
        trigger_opt.update(self.trigger_opt)
        out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                 trace_opt=trace_opt)
        self.dataset_clusterer_dict['global_fwd'] = out[0]
        self.intermediates_dict['global_fwd'] = out[1]

        trigger_dict = {'raw_rev': BehaviorCodes.REV, 'raw_fwd': BehaviorCodes.FWD,
                        'raw_vt': BehaviorCodes.VENTRAL_TURN, 'raw_dt': BehaviorCodes.DORSAL_TURN}
        trace_opt = dict(residual_mode=None)
        trace_opt.update(trace_base_opt)
        # Raw reversal triggered and forward triggered
        for trigger_type, state in trigger_dict.items():
            try:
                trigger_opt = dict(use_hilbert_phase=False, state=state)
                trigger_opt.update(self.trigger_opt)
                out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                         trace_opt=trace_opt)
                self.dataset_clusterer_dict[trigger_type] = out[0]
                self.intermediates_dict[trigger_type] = out[1]
            except (IndexError, KeyError):
                print(f"Trigger type {trigger_type} failed; this may be because the data is immobilized")

        # Optional
        if self.calc_stimulus:
            trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.STIMULUS)
            trigger_opt.update(self.trigger_opt)
            out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                     trace_opt=trace_opt)
            self.dataset_clusterer_dict['stimulus'] = out[0]
            self.intermediates_dict['stimulus'] = out[1]

    def get_clusterer_from_trigger_type(self, trigger_type):
        trigger_mapping = self.dataset_clusterer_dict
        if trigger_type not in trigger_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(trigger_mapping.keys())}')
        return trigger_mapping[trigger_type]

    def get_df_triggered_from_trigger_type(self, trigger_type, return_individual_traces=False):
        """
        Returns by default a precalculated dataframe of the form:
        - Columns: Neuron names combined with the dataset name (e.g. 2022-11-23_worm9_BAGL)
        - Rows: Time points

        Parameters
        ----------
        trigger_type
        return_individual_traces - if False, then return the average across events per dataset per neuron (default)
            if True, then return a dictionary of dataframes, where each dataframe is each event within the dataset.
            Note: each dataset has a different number of events (rows), but the same number of time points (columns)

        Returns
        -------

        """
        df_mapping = self.intermediates_dict
        if not return_individual_traces:
            df_mapping = {k: v[1] if v else None for k, v in df_mapping.items()}
        else:
            df_mapping = {k: v[2] if v else None for k, v in df_mapping.items()}
        if trigger_type not in df_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(df_mapping.keys())}')
        return df_mapping[trigger_type]

    def get_title_from_trigger_type(self, trigger_type):
        title_mapping = {'raw_rev': 'Raw reversal triggered',
                         'raw_fwd': 'Raw forward triggered',
                         'raw_vt': 'Raw ventral turn triggered',
                         'raw_dt': 'Raw dorsal turn triggered',
                         'global_rev': 'Global reversal triggered',
                         'global_fwd': 'Global forward triggered',
                         'residual': 'Residual undulation triggered',
                         'residual_collision': 'Residual collision triggered',
                         'residual_rectified_fwd': 'Residual (rectified fwd, undulation triggered)',
                         'residual_rectified_rev': 'Residual (rectified rev, undulation triggered)',
                         'kymo': 'Kymograph',
                         'stimulus': 'Stimulus triggered'}
        if trigger_type not in title_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(title_mapping.keys())}')
        return title_mapping[trigger_type]

    def get_trace_difference_auc(self, trigger_type, neuron0, neuron1, num_iters=100, z_score=False,
                                 norm_type='corr', shuffle_dataset_pairs=True, return_individual_traces=False):
        """
        Calculates the area under the curve of the difference between two neurons.

        Parameters
        ----------
        trigger_type
        neuron0
        neuron1
        num_iters - number of iterations to perform (only used if shuffle_dataset_pairs is True)
        z_score - if True, then we will z-score the traces before calculating the difference
        shuffle_dataset_pairs - if True, then we will shuffle the dataset pairs and subtract them. If False, then we
            will only subtract the same dataset, meaning that both neurons must be present.
        return_individual_traces - if True, then return a list of the difference for each event in each dataset.
            If False, then return a list of the difference for each dataset. (default)
            Mutually exclusive with shuffle_dataset_pairs=True

        Returns
        -------

        """
        df_or_dict = self.get_df_triggered_from_trigger_type(trigger_type,
                                                             return_individual_traces=return_individual_traces)
        if z_score:
            if not return_individual_traces:
                df_or_dict = (df_or_dict - df_or_dict.mean()) / df_or_dict.std()
            else:
                # Then each dataset should be z-scored separately, but across all events simultaneously
                # Numpy will do this automatically (pandas tries to keep the axis)
                keys_to_remove = []
                for _name, df in df_or_dict.items():
                    if df is None:
                        keys_to_remove.append(_name)
                        continue
                    df_or_dict[_name] = (df - np.nanmean(df)) / np.nanstd(df)
                for k in keys_to_remove:
                    del df_or_dict[k]

        if not return_individual_traces:
            names0 = [n for n in list(df_or_dict.columns) if neuron0 in n]
            names1 = [n for n in list(df_or_dict.columns) if neuron1 in n]
        else:
            names0 = [n for n in list(df_or_dict.keys()) if neuron0 in n]
            names1 = [n for n in list(df_or_dict.keys()) if neuron1 in n]
        if len(names0) == 0 or len(names1) == 0:
            raise ValueError(f'Neuron name {neuron0} or {neuron1} not found')

        # Define the summary statistic
        def norm(x, y):
            if norm_type == 'corr':
                # Calculate the correlation, which is what is shown in the clustered matrix
                if isinstance(x, pd.Series):
                    return np.nanmean(x.corr(y))
                else:
                    # Then need to apply the correlation to each row (paired)
                    ind = x.index
                    all_corrs = [x.loc[i, :].corr(y.loc[i, :]) for i in ind]
                    return np.nanmean(all_corrs)
            elif norm_type == 'auc':
                # This is the same as the mean squared difference
                delta = np.nanmean((x - y).pow(2))
                x_norm = np.nanmean(x.pow(2))
                y_norm = np.nanmean(y.pow(2))
                return delta / (x_norm + y_norm)
            else:
                raise ValueError(f"Invalid norm type {norm_type}; must be one of 'corr' or 'auc'")

        if shuffle_dataset_pairs:
            if return_individual_traces:
                raise NotImplementedError("Shuffling dataset pairs not implemented for return_individual_traces=True,"
                                          " because there is no clean way to pair up the traces")
            # There are two lists of neurons, and we want to choose random pairs
            samples = [(random.choice(names0), random.choice(names1)) for _ in range(num_iters)]
            all_norms = []
            for name0, name1 in samples:
                trace0 = df_or_dict[name0]
                trace1 = df_or_dict[name1]
                all_norms.append(norm(trace0, trace1))
        else:
            # Get the names of the datasets
            if not return_individual_traces:
                column_names = get_names_from_df(df_or_dict)
            else:
                column_names = list(df_or_dict.keys())
            split_column_names = split_flattened_index(column_names)
            all_dataset_names = {dataset_name for col_name, (dataset_name, neuron_name) in split_column_names.items()}
            # Loop through datasets, and check if both neurons are present
            all_norms = []
            for dataset_name in all_dataset_names:
                name0 = f"{dataset_name}_{neuron0}"
                name1 = f"{dataset_name}_{neuron1}"
                if name0 in column_names and name1 in column_names:
                    trace0 = df_or_dict[name0]
                    trace1 = df_or_dict[name1]
                    # Regardless of the traces being one trace or a dataframe of events, we compress to a single norm
                    all_norms.append(norm(trace0, trace1))
                else:
                    all_norms.append(np.nan)
            if len(all_norms) == 0:
                raise ValueError(f"Neurons {neuron0} and {neuron1} not found simultaneously in any datasets")

        return all_norms

    def get_trace_difference_auc_multiple_neurons(self, trigger_type, list_of_neurons, norm_type='corr',
                                                  baseline_neuron=None, df_norms=None, **kwargs):
        """
        Use get_trace_difference for pairs of neurons, generated as all combinations of the neurons in list_of_neurons.

        Parameters
        ----------
        trigger_type
        list_of_neurons

        Returns
        -------

        """
        if baseline_neuron is None:
            neuron_combinations = list(itertools.combinations(list_of_neurons, 2))
        else:
            neuron_combinations = [(baseline_neuron, n) for n in list_of_neurons if n != baseline_neuron]
        dict_norms = {}
        for neuron0, neuron1 in tqdm(neuron_combinations, leave=False):
            key = f"{neuron0}-{neuron1}"
            dict_norms[key] = self.get_trace_difference_auc(trigger_type, neuron0, neuron1, norm_type=norm_type,
                                                            **kwargs)

        df_norms = pd.DataFrame(dict_norms)
        return df_norms

    def get_fig_opt(self, height_factor=1, width_factor=1):
        return dict(dpi=300, figsize=(width_factor*10/3, height_factor*10/(2*3)))

    def plot_triggered_average_single_neuron(self, neuron_name, trigger_type, output_folder=None,
                                             fig=None, ax=None, title=None, include_neuron_in_title=False,
                                             xlim=None, ylim=None, min_lines=2,
                                             show_title=False,
                                             color=None, z_score=False, fig_kwargs=None, legend=False, i_figure=3,
                                             DEBUG=False):
        if fig_kwargs is None:
            fig_kwargs = {}
        if color is None:
            color = self.get_color_from_trigger_type(trigger_type)
        df_subset = self.get_traces_single_neuron(trigger_type, neuron_name, DEBUG)

        if df_subset.shape[1] == 0:
            print(f"Neuron name {neuron_name} not found, skipping")
            return

        # Plot the triggered average for each neuron
        if ax is None:
            fig_opt_trigger = self.get_fig_opt(**fig_kwargs)
            fig, ax = plt.subplots(**fig_opt_trigger)
            is_second_plot = False
        else:
            is_second_plot = True
        if z_score:
            df_subset = (df_subset - df_subset.mean()) / df_subset.std()
        df_subset = df_subset.T

        min_lines = min(min_lines, df_subset.shape[1])
        if DEBUG:
            print('df_subset', df_subset)
        plot_triggered_average_from_matrix_low_level(df_subset, 0, min_lines, show_individual_lines=False,
                                                     is_second_plot=is_second_plot, ax=ax,
                                                     color=color, label=neuron_name, show_horizontal_line=False)
        if 'rectified_rev' in trigger_type:
            behavior_shading_type = 'both'
        elif 'rectified_fwd' in trigger_type:
            behavior_shading_type = None
        elif 'rev' in trigger_type:
            behavior_shading_type = 'rev'
        elif 'fwd' in trigger_type:
            behavior_shading_type = 'fwd'
        else:
            behavior_shading_type = None
        if behavior_shading_type is not None:
            index_conversion = df_subset.columns
            try:
                shade_triggered_average(ind_preceding=20, index_conversion=index_conversion,
                                        behavior_shading_type=behavior_shading_type, ax=ax)
            except IndexError:
                print(f"Index error for {neuron_name} and {trigger_type}; skipping shading")

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if z_score:
            plt.ylabel("Amplitude (z-scored)")
        else:
            plt.ylabel("$\Delta R / R_{50}$")
        if show_title:
            if title is None:
                title = self.get_title_from_trigger_type(trigger_type)
                plt.title(f"{neuron_name} (n={df_subset.shape[1]}) {title}")
            else:
                if include_neuron_in_title:
                    plt.title(f"{neuron_name} {title}")
                else:
                    plt.title(title)
        else:
            plt.title("")
        plt.xlabel("Time (s)")
        if legend:
            plt.legend()
        plt.tight_layout()

        if output_folder is not None:
            if i_figure == 3:
                fig_opt = dict(width_factor=0.5, height_factor=0.25)
            elif i_figure > 3:
                if 'rectified' in trigger_type:
                    fig_opt = dict(width_factor=0.35, height_factor=0.15)
                else:
                    fig_opt = dict(width_factor=0.25, height_factor=0.15)
            else:
                raise NotImplementedError(f"i_figure={i_figure} not implemented")
            apply_figure_settings(fig, plotly_not_matplotlib=False, **fig_opt)

            title = self.get_title_from_trigger_type(trigger_type)
            fname = title.replace(" ", "_").replace(",", "").lower()
            fname = os.path.join(output_folder, f'{neuron_name}-{fname}.png')
            plt.savefig(fname, transparent=True)
            plt.savefig(fname.replace(".png", ".svg"))

        return fig, ax

    def get_traces_single_neuron(self, trigger_type, neuron_name, DEBUG=False):
        df = self.get_df_triggered_from_trigger_type(trigger_type)
        # Get the full names of all the neurons with this name
        # Names will be like '2022-11-23_worm9_BAGL' and we are checking for 'BAGL'
        neuron_names = [n for n in list(df.columns) if neuron_name in n]
        if DEBUG:
            print(f"Found {len(neuron_names)} neurons with name {neuron_name}")
            print(f"Neuron names: {neuron_names}")
        df_subset = df.loc[:, neuron_names]
        return df_subset

    def plot_triggered_average_multiple_neurons(self, neuron_list, trigger_type, color_list=None,
                                                output_folder=None, **kwargs):
        """
        Uses plot_triggered_average_single_neuron to plot multiple neurons on the same plot.

        Parameters
        ----------
        neuron_list
        trigger_type
        color_list
        title
        output_folder

        Returns
        -------

        """
        if color_list is None:
            # They will all be the same color
            color_list = [self.get_behavior_color_from_neuron_name(n) for n in neuron_list]

        fig, ax = None, None
        for i, (neuron, color) in enumerate(zip(neuron_list, color_list)):
            # Only set the output folder for the last neuron
            if i == len(neuron_list) - 1:
                this_output_folder = output_folder
            else:
                this_output_folder = None
            fig, ax = self.plot_triggered_average_single_neuron(neuron, trigger_type, output_folder=this_output_folder,
                                                                include_neuron_in_title=False, ax=ax, fig=fig,
                                                                fig_kwargs=dict(height_factor=2),
                                                                color=color, **kwargs)
        return fig, ax


@dataclass
class PaperExampleTracePlotter(PaperColoredTracePlotter):
    """
    For plotting example traces, specifically a stack of 3 traces:
    - Raw
    - Global
    - Residual
    """

    project: ProjectData

    xlim: Optional[tuple] = (0, 150)
    ylim: Optional[tuple] = None

    def __post_init__(self):
        self.project.use_physical_time = True

        # Load the cache
        _ = self.df_traces
        _ = self.df_traces_global
        _ = self.df_traces_residual

    @property
    def df_traces(self):
        return self.project.calc_paper_traces()

    @property
    def df_traces_residual(self):
        return self.project.calc_paper_traces_residual()

    @property
    def df_traces_global(self):
        return self.project.calc_paper_traces_global()

    def get_figure_opt(self):
        return dict(dpi=300, figsize=(10/3, 10/2), gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

    def plot_triple_traces(self, neuron_name, title=False, legend=False,
                           output_foldername=None, **kwargs):
        """
        Plot the three traces (raw, global, residual) on the same plot.
        If output_foldername is not None, save the plot in that folder.

        Parameters
        ----------
        neuron_name
        output_foldername

        Returns
        -------

        """
        df_traces = self.df_traces
        df_traces_global = self.df_traces_global
        df_traces_residual = self.df_traces_residual

        fig_opt = self.get_figure_opt()
        fig, axes = plt.subplots(**fig_opt, nrows=3, ncols=1)
        xlim = kwargs.get('xlim', self.xlim)
        ylim = kwargs.get('ylim', self.ylim)

        # Do all on one plot
        trace_dict = {'Raw': (df_traces[neuron_name], self.get_color_from_trigger_type('raw')),
                      'Global': (df_traces_global[neuron_name], self.get_color_from_trigger_type('global')),
                      'Residual': (df_traces_residual[neuron_name], self.get_color_from_trigger_type('residual'))}

        for i, (name, vals) in enumerate(trace_dict.items()):
            # Original trace
            ax = axes[i]
            ax.plot(vals[0], color=vals[1], label=name)
            if title and i == 0:
                ax.set_title(neuron_name)
            if legend:
                ax.legend(frameon=False)
            ax.set_ylabel(r"$\Delta R / R_{50}$")
            ax.set_xlim(xlim)
            ax.autoscale(enable=True, axis='y')  # Scale to the actually visible data (leaving x as set)
            if ylim is None:
                # If no given ylim, use the first trace's ylim
                ylim = ax.get_ylim()
            else:
                ax.set_ylim(ylim)

            if i < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (s)")
            self.project.shade_axis_using_behavior(ax)

        # Remove space between subplots
        plt.subplots_adjust(hspace=0)

        apply_figure_settings(fig, width_factor=0.25, height_factor=0.3, plotly_not_matplotlib=False)

        if output_foldername:
            fname = os.path.join(output_foldername, f'{neuron_name}-combined_traces.png')
            plt.savefig(fname, transparent=True)
            fig.savefig(fname.replace(".png", ".svg"))
