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
from wbfm.utils.traces.triggered_averages import ax_plot_func_for_grid_plot
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
    opt = dict(channel_mode='ratio', calculation_mode='integration', to_save=False,
               color_using_behavior=False, share_y_axis=False, min_nonnan=0.8)
    for name, state in all_triggers.items():
        func = lambda *args, **kwargs: ax_plot_func_for_grid_plot(*args, project_data=proj_dat, state=state,
                                                                  min_lines=5,
                                                                  **kwargs)
        make_grid_plot_using_project(proj_dat, **opt, ax_plot_func=func)

        # Save in the project
        fname = f"{name}_triggered_average.png"
        fname = vis_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        plt.savefig(fname)