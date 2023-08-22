"""
Designed to plot the triggered average of the paper's datasets.
"""
import os
from dataclasses import dataclass
from typing import Dict

from matplotlib import pyplot as plt

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.traces.triggered_averages import clustered_triggered_averages_from_list_of_projects, \
    ClusteredTriggeredAverages, plot_triggered_average_from_matrix_low_level


@dataclass
class PaperMultiDatasetTriggeredAverage:
    """Class to plot the triggered average of the paper's datasets."""

    all_projects: Dict[str, ProjectData]

    # Three different sets of parameters: raw, global, and residual
    dataset_clusterer_raw: ClusteredTriggeredAverages = None
    dataset_clusterer_global: ClusteredTriggeredAverages = None
    dataset_clusterer_residual: ClusteredTriggeredAverages = None

    intermediates_raw = None
    intermediates_global = None
    intermediates_residual = None

    # Use these to build rectified (single-state only) triggered averages
    # For now, only need the residual ones
    dataset_clusterer_residual_rectified_fwd: ClusteredTriggeredAverages = None
    dataset_clusterer_residual_rectified_rev: ClusteredTriggeredAverages = None

    intermediates_residual_rectified_fwd = None
    intermediates_residual_rectified_rev = None

    def __post_init__(self):
        # Analyze the project data to get the clusterers and intermediates
        trace_base_opt = dict(interpolate_nan=True, channel_mode='dr_over_r_50', remove_outliers=True,
                              rename_neurons_using_manual_ids=True, manual_id_confidence_threshold=0)

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

        # Slow (global)
        trigger_opt = dict(use_hilbert_phase=False, state=BehaviorCodes.REV)
        trace_opt = dict(residual_mode='pca_global')
        trace_opt.update(trace_base_opt)
        out = clustered_triggered_averages_from_list_of_projects(self.all_projects, trigger_opt=trigger_opt,
                                                                 trace_opt=trace_opt)
        self.dataset_clusterer_global = out[0]
        self.intermediates_global = out[1]

    def get_clusterer_from_trigger_type(self, trigger_type):
        trigger_mapping = {'raw': self.dataset_clusterer_raw,
                           'global': self.dataset_clusterer_global,
                           'residual': self.dataset_clusterer_residual,
                           'residual_rectified_fwd': self.dataset_clusterer_residual_rectified_fwd,
                           'residual_rectified_rev': self.dataset_clusterer_residual_rectified_rev}
        if trigger_type not in trigger_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(trigger_mapping.keys())}')
        return trigger_mapping[trigger_type]

    def get_df_triggered_from_trigger_type(self, trigger_type):
        df_mapping = {#'raw': self.intermediates_raw[1],  # TODO: implement raw
                      'global': self.intermediates_global[1],
                      'residual': self.intermediates_residual[1],
                      'residual_rectified_fwd': self.intermediates_residual_rectified_fwd[1],
                      'residual_rectified_rev': self.intermediates_residual_rectified_rev[1]}
        if trigger_type not in df_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(df_mapping.keys())}')
        return df_mapping[trigger_type]

    def get_color_from_trigger_type(self, trigger_type):
        color_mapping = {'raw': 'tab:blue',
                         'global': 'tab:orange',
                         'residual': 'tab:green',
                         'residual_rectified_fwd': 'tab:green',
                         'residual_rectified_rev': 'tab:green',
                         'kymo': 'black'}
        if trigger_type not in color_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(color_mapping.keys())}')
        return color_mapping[trigger_type]

    def get_title_from_trigger_type(self, trigger_type):
        title_mapping = {'raw': 'Raw',
                         'global': 'Global reversal triggered',
                         'residual': 'Residual undulation triggered',
                         'residual_rectified_fwd': 'Residual (rectified fwd, undulation triggered)',
                         'residual_rectified_rev': 'Residual (rectified rev, undulation triggered)',
                         'kymo': 'Kymograph'}
        if trigger_type not in title_mapping:
            raise ValueError(f'Invalid trigger type: {trigger_type}; must be one of {list(title_mapping.keys())}')
        return title_mapping[trigger_type]

    def get_fig_opt(self):
        return dict(dpi=300, figsize=(5, 5))

    def plot_triggered_average_single_neuron(self, neuron_name, trigger_type, output_foldername=None,
                                             ax=None,
                                             DEBUG=False):
        # clusterer = self.get_clusterer_from_trigger_type(trigger_type)
        color = self.get_color_from_trigger_type(trigger_type)
        df = self.get_df_triggered_from_trigger_type(trigger_type)
        title = self.get_title_from_trigger_type(trigger_type)

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
            fig_opt_trigger = self.get_fig_opt()
            fig, ax = plt.subplots(**fig_opt_trigger)
            is_second_plot = False
        else:
            is_second_plot = True

        # TODO: do not hardcode the ind_preceding
        df_subset = df.loc[:, neuron_names].T
        min_lines = min(3, len(neuron_names))
        if DEBUG:
            print(df_subset)
        plot_triggered_average_from_matrix_low_level(df_subset, 0, min_lines, False,
                                                     is_second_plot=is_second_plot, ax=ax,
                                                     color=color)
        plt.ylabel("dR/R50")
        plt.title(f"{neuron_name} (n={len(neuron_names)}) {title}")
        plt.xlabel("Time (s)")
        plt.tight_layout()

        if output_foldername is not None:
            fname = title.replace(" ", "_").replace(",", "").lower()
            fname = os.path.join(output_foldername, f'{fname}.png')
            plt.savefig(fname)
            plt.savefig(fname.replace(".png", ".svg"))

        return ax
