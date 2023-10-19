"""
Designed to plot the triggered average of the paper's datasets.
"""
import os
from dataclasses import dataclass
from typing import Dict, Optional

from matplotlib import pyplot as plt

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, shade_triggered_average
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.traces.triggered_averages import clustered_triggered_averages_from_list_of_projects, \
    ClusteredTriggeredAverages, plot_triggered_average_from_matrix_low_level


@dataclass
class PaperColoredTracePlotter:
    """
    Class to plot the colored traces of the paper's datasets.

    Specifically for raw/global/residual decompositions
    """

    def get_color_from_trigger_type(self, trigger_type):
        color_mapping = {'raw_rev': 'tab:blue',
                         'raw': 'tab:blue',
                         'raw_fwd': 'tab:blue',
                         'global_rev': 'tab:orange',
                         'global': 'tab:orange',
                         'global_fwd': 'tab:orange',
                         'residual': 'tab:green',
                         'residual_rectified_fwd': 'tab:green',
                         'residual_rectified_rev': 'tab:green',
                         'kymo': 'black'}
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
            RID='tab:red',
            RME='tab:purple',
            AVB='tab:red',
            RIB='tab:red',
        )
        # Add keys by adding the L/R and V/D suffixes
        for k in list(color_mapping.keys()):
            color_mapping[k + 'L'] = color_mapping[k]
            color_mapping[k + 'R'] = color_mapping[k]
            color_mapping[k + 'V'] = color_mapping[k]
            color_mapping[k + 'D'] = color_mapping[k]
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
    dataset_clusterer_raw_rev: ClusteredTriggeredAverages = None
    dataset_clusterer_raw_fwd: ClusteredTriggeredAverages = None
    dataset_clusterer_global_rev: ClusteredTriggeredAverages = None
    dataset_clusterer_global_fwd: ClusteredTriggeredAverages = None
    dataset_clusterer_residual: ClusteredTriggeredAverages = None

    intermediates_raw_rev = None
    intermediates_raw_fwd = None
    intermediates_global_rev = None
    intermediates_global_fwd = None
    intermediates_residual = None

    # Use these to build rectified (single-state only) triggered averages
    # For now, only need the residual ones
    dataset_clusterer_residual_rectified_fwd: ClusteredTriggeredAverages = None
    dataset_clusterer_residual_rectified_rev: ClusteredTriggeredAverages = None

    intermediates_residual_rectified_fwd = None
    intermediates_residual_rectified_rev = None

    def __post_init__(self):
        # Analyze the project data to get the clusterers and intermediates
        trace_base_opt = self.get_trace_opt(min_nonnan=self.min_nonnan)

        if self.calculate_residual:
            try:
                # Note: these won't work for immobilized data

                # Fast (residual)
                trigger_opt = dict(use_hilbert_phase=True, state=None)
                trace_opt = dict(residual_mode='pca')
                trace_opt.update(trace_base_opt)
                out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                         trace_opt=trace_opt)
                self.dataset_clusterer_residual = out[0]
                self.intermediates_residual = out[1]

                # Residual rectified forward
                trigger_opt['only_allow_events_during_state'] = BehaviorCodes.FWD
                cluster_opt = {}
                out = clustered_triggered_averages_from_list_of_projects(self.all_projects, cluster_opt=cluster_opt,
                                                                         trigger_opt=trigger_opt, trace_opt=trace_opt)
                self.dataset_clusterer_residual_rectified_fwd = out[0]
                self.intermediates_residual_rectified_fwd = out[1]

                # Residual rectified reverse
                trigger_opt['only_allow_events_during_state'] = BehaviorCodes.REV
                cluster_opt = {}
                out = clustered_triggered_averages_from_list_of_projects(self.all_projects, cluster_opt=cluster_opt,
                                                                         trigger_opt=trigger_opt, trace_opt=trace_opt)
                self.dataset_clusterer_residual_rectified_rev = out[0]
                self.intermediates_residual_rectified_rev = out[1]

            except TypeError:
                print("Hilbert triggered averages failed; this may be because the data is immobilized")
                print("Only 'global' triggered averages will be available")

        # Slow reversal triggered (global)
        trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.REV)
        trace_opt = dict(residual_mode='pca_global')
        trace_opt.update(trace_base_opt)
        out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                 trace_opt=trace_opt)
        self.dataset_clusterer_global_rev = out[0]
        self.intermediates_global_rev = out[1]

        # Slow forward triggered (global)
        trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.FWD)
        out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                 trace_opt=trace_opt)
        self.dataset_clusterer_global_fwd = out[0]
        self.intermediates_global_fwd = out[1]

        # Raw reversal triggered and forward triggered
        trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.REV)
        trace_opt = dict(residual_mode=None)
        trace_opt.update(trace_base_opt)
        out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                 trace_opt=trace_opt)
        self.dataset_clusterer_raw_rev = out[0]
        self.intermediates_raw_rev = out[1]

        trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.FWD)
        out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                 trace_opt=trace_opt)
        self.dataset_clusterer_raw_fwd = out[0]
        self.intermediates_raw_fwd = out[1]

    def get_clusterer_from_trigger_type(self, trigger_type):
        trigger_mapping = {'raw_rev': self.dataset_clusterer_raw_rev,
                           'raw_fwd': self.dataset_clusterer_raw_fwd,
                           'global_rev': self.dataset_clusterer_global_rev,
                           'global_fwd': self.dataset_clusterer_global_fwd,
                           'residual': self.dataset_clusterer_residual,
                           'residual_rectified_fwd': self.dataset_clusterer_residual_rectified_fwd,
                           'residual_rectified_rev': self.dataset_clusterer_residual_rectified_rev}
        if trigger_type not in trigger_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(trigger_mapping.keys())}')
        return trigger_mapping[trigger_type]

    def get_df_triggered_from_trigger_type(self, trigger_type):
        df_mapping = {'raw_rev': self.intermediates_raw_rev,
                      'raw_fwd': self.intermediates_raw_fwd,
                      'global_rev': self.intermediates_global_rev,
                      'global_fwd': self.intermediates_global_fwd,
                      'residual': self.intermediates_residual,
                      'residual_rectified_fwd': self.intermediates_residual_rectified_fwd,
                      'residual_rectified_rev': self.intermediates_residual_rectified_rev}
        df_mapping = {k: v[1] if v else None for k, v in df_mapping.items()}
        if trigger_type not in df_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(df_mapping.keys())}')
        return df_mapping[trigger_type]

    def get_title_from_trigger_type(self, trigger_type):
        title_mapping = {'raw_rev': 'Raw reversal triggered',
                         'raw_fwd': 'Raw forward triggered',
                         'global_rev': 'Global reversal triggered',
                         'global_fwd': 'Global forward triggered',
                         'residual': 'Residual undulation triggered',
                         'residual_rectified_fwd': 'Residual (rectified fwd, undulation triggered)',
                         'residual_rectified_rev': 'Residual (rectified rev, undulation triggered)',
                         'kymo': 'Kymograph'}
        if trigger_type not in title_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(title_mapping.keys())}')
        return title_mapping[trigger_type]

    def get_fig_opt(self, height_factor=1, width_factor=1):
        return dict(dpi=300, figsize=(width_factor*10/3, height_factor*10/(2*3)))

    def plot_triggered_average_single_neuron(self, neuron_name, trigger_type, output_folder=None,
                                             ax=None, title=None, include_neuron_in_title=True, xlim=None,
                                             color=None, z_score=False, fig_kwargs=None, legend=False, DEBUG=False):
        if fig_kwargs is None:
            fig_kwargs = {}
        if color is None:
            color = self.get_color_from_trigger_type(trigger_type)
        df = self.get_df_triggered_from_trigger_type(trigger_type)

        # Get the full names of all the neurons with this name
        # Names will be like '2022-11-23_worm9_BAGL' and we are checking for 'BAGL'
        neuron_names = [n for n in list(df.columns) if neuron_name in n]
        if DEBUG:
            print(f"Found {len(neuron_names)} neurons with name {neuron_name}")
            print(f"Neuron names: {neuron_names}")
        if len(neuron_names) == 0:
            print(f"Neuron name {neuron_name} not found, skipping")
            return

        # Plot the triggered average for each neuron
        if ax is None:
            fig_opt_trigger = self.get_fig_opt(**fig_kwargs)
            fig, ax = plt.subplots(**fig_opt_trigger)
            is_second_plot = False
        else:
            is_second_plot = True

        df_subset = df.loc[:, neuron_names]
        if z_score:
            df_subset = (df_subset - df_subset.mean()) / df_subset.std()
        df_subset = df_subset.T

        min_lines = min(3, len(neuron_names))
        if DEBUG:
            print(df_subset)
        plot_triggered_average_from_matrix_low_level(df_subset, 0, min_lines, False,
                                                     is_second_plot=is_second_plot, ax=ax,
                                                     color=color, label=neuron_name)
        if 'residual' in trigger_type or 'rectified' in trigger_type:
            behavior_shading_type = None
        elif 'rev' in trigger_type:
            behavior_shading_type = 'rev'
        elif 'fwd' in trigger_type:
            behavior_shading_type = 'fwd'
        else:
            behavior_shading_type = None
        if behavior_shading_type is not None:
            index_conversion = df_subset.columns
            shade_triggered_average(ind_preceding=20, index_conversion=index_conversion,
                                    behavior_shading_type=behavior_shading_type, ax=ax)
        if xlim is not None:
            ax.set_xlim(xlim)
        if z_score:
            plt.ylabel("Amplitude (z-scored)")
        else:
            plt.ylabel("dR/R50")
        if title is None:
            title = self.get_title_from_trigger_type(trigger_type)
            plt.title(f"{neuron_name} (n={len(neuron_names)}) {title}")
        else:
            if include_neuron_in_title:
                plt.title(f"{neuron_name} {title}")
            else:
                plt.title(title)
        plt.xlabel("Time (s)")
        if legend:
            plt.legend()
        plt.tight_layout()

        if output_folder is not None:
            title = self.get_title_from_trigger_type(trigger_type)
            fname = title.replace(" ", "_").replace(",", "").lower()
            fname = os.path.join(output_folder, f'{neuron_name}-{fname}.png')
            plt.savefig(fname)
            plt.savefig(fname.replace(".png", ".svg"))

        return ax

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

        ax = None
        for i, (neuron, color) in enumerate(zip(neuron_list, color_list)):
            # Only set the output folder for the last neuron
            if i == len(neuron_list) - 1:
                this_output_folder = output_folder
            else:
                this_output_folder = None
            ax = self.plot_triggered_average_single_neuron(neuron, trigger_type, output_folder=this_output_folder,
                                                           include_neuron_in_title=False, ax=ax,
                                                           fig_kwargs=dict(height_factor=2),
                                                           color=color, **kwargs)


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
        self.project.use_physical_x_axis = True

        # Load the cache
        _ = self.df_traces
        _ = self.df_traces_global
        _ = self.df_traces_residual

    @property
    def df_traces(self):
        trace_opt = self.get_trace_opt()
        return self.project.calc_default_traces(**trace_opt)

    @property
    def df_traces_residual(self):
        trace_opt = self.get_trace_opt()
        trace_opt['residual_mode'] = 'pca'
        return self.project.calc_default_traces(**trace_opt)

    @property
    def df_traces_global(self):
        trace_opt = self.get_trace_opt()
        trace_opt['residual_mode'] = 'pca_global'
        return self.project.calc_default_traces(**trace_opt)

    def get_figure_opt(self):
        return dict(dpi=300, figsize=(10/3, 10/2))

    def plot_triple_traces(self, neuron_name, output_foldername=None, **kwargs):
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
        trace_dict = {'Original trace': (df_traces[neuron_name], self.get_color_from_trigger_type('raw')),
                      'Global component': (df_traces_global[neuron_name], self.get_color_from_trigger_type('global')),
                      'Residual component': (df_traces_residual[neuron_name], self.get_color_from_trigger_type('residual'))}

        for i, (name, vals) in enumerate(trace_dict.items()):
            # Original trace
            ax = axes[i]
            ax.plot(vals[0], color=vals[1])
            ax.set_title(name)
            ax.set_ylabel("dR/R50")
            ax.set_xlim(xlim)
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

        plt.tight_layout()

        if output_foldername:
            fname = os.path.join(output_foldername, f'{neuron_name}-combined_traces.png')
            plt.savefig(fname)
            fig.savefig(fname.replace(".png", ".svg"))
