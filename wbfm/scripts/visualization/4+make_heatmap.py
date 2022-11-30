"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import TinyDbObserver

# main function
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.visualization.plot_traces import make_heatmap_using_project

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
    proj_dat.verbose = 0

    make_heatmap_using_project(proj_dat)
