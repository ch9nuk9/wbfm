"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

# Experiment tracking
import logging

import sacred
from sacred import Experiment
from sacred import SETTINGS
# main function
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch

from DLC_for_WBFM.utils.pipeline.traces_pipeline import get_traces_from_3d_tracks_using_config, \
    extract_traces_using_config
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.visualization.plot_traces import make_grid_plot_from_project

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    traces_cfg = cfg.get_traces_config()

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def make_full_tracks(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    trace_cfg = _config['traces_cfg']
    project_cfg = _config['cfg']

    with safe_cd(_config['project_dir']):
        # Reads masks from disk, and writes traces
        extract_traces_using_config(project_cfg, trace_cfg, name_mode='neuron', DEBUG=DEBUG)

        # By default make some visualizations
        # Note: reloads the project data
        logging.info("Making default grid plots")
        proj_dat = ProjectData.load_final_project_data_from_config(project_cfg)
        make_grid_plot_from_project(proj_dat, channel_mode='all', calculation_mode='integration')
