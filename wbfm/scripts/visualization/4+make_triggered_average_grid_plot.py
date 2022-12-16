"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.visualization.plot_traces import make_default_triggered_average_plots, \
    make_pirouette_split_triggered_average_plots

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

    make_default_triggered_average_plots(_config['project_path'])

    make_pirouette_split_triggered_average_plots(_config['project_path'])
