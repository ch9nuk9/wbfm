"""
Designed to plot the triggered average of the paper's datasets.
"""

from dataclasses import dataclass
from typing import Dict

from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.traces.triggered_averages import clustered_triggered_averages_from_list_of_projects


@dataclass
class PaperMultiDatasetTriggeredAverage:
    """Class to plot the triggered average of the paper's datasets."""

    all_projects: Dict[str, ProjectData]

    # Three different sets of parameters: raw, global, and residual
    dataset_clusterer_raw = None
    dataset_clusterer_global = None
    dataset_clusterer_residual = None

    intermediates_raw = None
    intermediates_global = None
    intermediates_residual = None

    # Use these to build rectified (single-state only) triggered averages
    # For now, only need the residual ones
    dataset_clusterer_residual_rectified_fwd = None
    dataset_clusterer_residual_rectified_rev = None

    intermediates_residual_rectified_fwd = None
    intermediates_residual_rectified_rev = None

    def __post_init__(self):
        # Analyze the project data to get the clusterers and intermediates
        trace_base_opt = dict(interpolate_nan=True, channel_mode='dr_over_r_50', remove_outliers=True,
                              rename_neurons_using_manual_ids=True)

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

    def plot_triggered_average_single_neuron(self, neuron_name, trigger_type):
        pass