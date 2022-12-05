"""
main
"""

from pathlib import Path

import matplotlib.pyplot as plt
# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import TinyDbObserver

# main function
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.traces.triggered_averages import ax_plot_func_for_grid_plot, FullDatasetTriggeredAverages
from wbfm.utils.visualization.plot_traces import make_grid_plot_using_project

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None)


@ex.config
def cfg(project_path):
    project_dir = str(Path(project_path).parent)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    proj_dat = ProjectData.load_final_project_data_from_config(_config['project_path'])
    vis_cfg = proj_dat.project_config.get_visualization_config()
    proj_dat.verbose = 0

    all_triggers = dict(reversal=1, forward=0)

    # All triggers
    trace_opt = dict(channel_mode='ratio', calculation_mode='integration', to_save=False,
                     color_using_behavior=False, share_y_axis=False, min_nonnan=0.8)
    df = proj_dat.calc_default_traces(**trace_opt)
    trigger_opt = dict(min_lines=5, ind_preceding=20, state=None, trace_len=df.shape[0])
    min_significant = 20
    ind_class = proj_dat.worm_posture_class.calc_triggered_average_indices(**trigger_opt)
    triggered_averages_class = FullDatasetTriggeredAverages(df, ind_class, min_points_for_significance=min_significant)

    for name, state in all_triggers.items():
        # Change option within class
        triggered_averages_class.ind_class.behavioral_state = state

        # First, general gridplot
        func = triggered_averages_class.ax_plot_func_for_grid_plot
        make_grid_plot_using_project(proj_dat, **trace_opt, ax_plot_func=func)

        fname = vis_cfg.resolve_relative_path(f"{name}_triggered_average.png", prepend_subfolder=True)
        plt.savefig(fname)

        # Second, gridplot with "significant" points marked
        func = triggered_averages_class.ax_plot_func_for_grid_plot
        make_grid_plot_using_project(proj_dat, **trace_opt, ax_plot_func=func)

        fname = vis_cfg.resolve_relative_path(f"{name}_triggered_average_significant_points_marked.png", prepend_subfolder=True)
        plt.savefig(fname)

        # Finally, a smaller subset of the grid plot (only neurons with enough signficant points)
        subset_neurons = triggered_averages_class.which_neurons_are_significant()
        func = triggered_averages_class.ax_plot_func_for_grid_plot
        make_grid_plot_using_project(proj_dat, **trace_opt, ax_plot_func=func, neuron_names_to_plot=subset_neurons)

        fname = vis_cfg.resolve_relative_path(f"{name}_triggered_average_neuron_subset.png", prepend_subfolder=True)
        plt.savefig(fname)
