"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
# main function
from sacred.observers import TinyDbObserver

import DLC_for_WBFM.utils.projects.monkeypatch_json
from DLC_for_WBFM.utils.pipeline.traces_pipeline import get_traces_from_3d_tracks_using_config
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config
from DLC_for_WBFM.utils.projects.utils_project import safe_cd

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = modular_project_config(project_path)
    project_dir = cfg.project_dir

    seg_cfg = cfg.get_segmentation_config()
    training_cfg = cfg.get_training_config()
    tracking_cfg = cfg.get_tracking_config()

    if not DEBUG:
        log_dir = cfg.get_log_dir()
        ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def make_full_tracks(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    seg_cfg = _config['seg_cfg']
    track_cfg = _config['track_cfg']
    trace_cfg = _config['trace_cfg']
    project_cfg = _config['project_cfg']

    with safe_cd(_config['project_dir']):
        get_traces_from_3d_tracks_using_config(seg_cfg,
                                               track_cfg,
                                               trace_cfg,
                                               project_cfg,
                                               DEBUG=DEBUG)
