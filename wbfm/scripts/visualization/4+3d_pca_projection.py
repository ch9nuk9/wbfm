"""
Plotting
"""

from pathlib import Path

# Experiment tracking
import sacred
from matplotlib import pyplot as plt
from sacred import Experiment
from sacred.observers import TinyDbObserver

# main function
from wbfm.utils.projects.finished_project_data import ProjectData, plot_pca_projection_3d_from_project

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, t_start=None, t_end=None, include_subplot=True)


@ex.config
def cfg(project_path):
    pass

@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    trace_kwargs = dict(channel_mode='dr_over_r_20', min_nonnan=0.9, filter_mode='rolling_mean')
    project_data = ProjectData.load_final_project_data_from_config(_config['project_path'])
    cfg = _config.copy()
    del cfg['project_path']
    del cfg['seed']
    plot_pca_projection_3d_from_project(project_data, trace_kwargs=trace_kwargs, **cfg)
    plt.show()
