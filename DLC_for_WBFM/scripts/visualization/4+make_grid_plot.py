"""
The top level function for producing dlc tracks in 3d
"""

from pathlib import Path
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
# Experiment tracking
import sacred
from DLC_for_WBFM.utils.visualization.plot_traces import make_grid_plot_from_project
from sacred import Experiment

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, trace_mode='green')


@ex.config
def cfg(project_path):
    project_dir = Path(project_path).parent
    project_cfg = load_config(project_path)

    traces_fname = Path(project_cfg['subfolder_configs']['traces'])
    traces_fname = Path(project_dir).joinpath(traces_fname)
    traces_cfg = dict(load_config(traces_fname))

@ex.automain
def make_dlc_labeled_videos(_config, _run):
    sacred.commands.print_config(_run)

    with safe_cd(_config['project_dir']):
        make_grid_plot_from_project(_config['traces_cfg'], trace_mode=_config['trace_mode'])
