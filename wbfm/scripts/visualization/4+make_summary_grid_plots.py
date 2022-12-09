"""
main
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import TinyDbObserver

# main function
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.visualization.plot_traces import make_default_summary_plots_using_config

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, channel_mode='all', calculation_mode='integration')


@ex.config
def cfg(project_path):
    project_dir = str(Path(project_path).parent)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    project_cfg = ModularProjectConfig(_config['project_path'])
    make_default_summary_plots_using_config(project_cfg)
