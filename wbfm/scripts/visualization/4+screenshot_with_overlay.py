"""
The top level function for producing dlc tracks in 3d
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import TinyDbObserver

# main function
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.visualization.napari_from_project_data_class import take_screenshot_using_project
from wbfm.utils.visualization.plot_traces import make_grid_plot_from_project

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

    project_data = ProjectData.load_final_project_data_from_config(_config['project_path'])
    project_data.verbose = 0

    additional_layers = [[('heatmap', 'count_nonnan')],
                         [('heatmap', 'std_of_green')],
                         [('heatmap', 'std_of_red')],
                         [('heatmap', 'max_of_green')],
                         [('heatmap', 'max_of_red')],
                         [('heatmap', 'max_of_ratio')],
                         [('heatmap', 'std_of_ratio')]]
    base_layers = ['Red data', 'Neuron IDs']

    take_screenshot_using_project(project_data, base_layers=base_layers, additional_layers=additional_layers)
