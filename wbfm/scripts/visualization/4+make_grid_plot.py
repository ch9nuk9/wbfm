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
from wbfm.utils.visualization.plot_traces import make_grid_plot_using_project

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, channel_mode='all', calculation_mode='integration')


@ex.config
def cfg(project_path):
    project_dir = str(Path(project_path).parent)

    # log_dir = str(Path(project_dir).joinpath('log'))
    # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    proj_dat = ProjectData.load_final_project_data_from_config(_config['project_path'])
    proj_dat.verbose = 0

    with safe_cd(_config['project_dir']):
        make_grid_plot_using_project(proj_dat, channel_mode=_config['channel_mode'],
                                     calculation_mode=_config['calculation_mode'])
