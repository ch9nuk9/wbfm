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
ex.add_config(project_path=None, t_end=None)


@ex.config
def cfg(project_path):
    project_dir = str(Path(project_path).parent)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    project_data = ProjectData.load_final_project_data_from_config(_config['project_path'])
    plot_pca_projection_3d_from_project(project_data, trace_kwargs={'filter_mode': 'rolling_mean'})
    plt.show()
