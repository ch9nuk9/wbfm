"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
# main function
from sacred.observers import TinyDbObserver
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step

from wbfm.pipeline.traces import match_segmentation_and_tracks_using_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project import safe_cd
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    seg_cfg = cfg.get_segmentation_config()
    tracking_cfg = cfg.get_tracking_config()
    traces_cfg = cfg.get_traces_config()

    use_training = tracking_cfg.config['leifer_params']['use_multiple_templates']
    check_all_needed_data_for_step(cfg, 4, training_data_required=use_training)

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def make_full_tracks(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    seg_cfg = _config['seg_cfg']
    track_cfg = _config['tracking_cfg']
    trace_cfg = _config['traces_cfg']
    project_cfg = _config['cfg']

    with safe_cd(_config['project_dir']):
        match_segmentation_and_tracks_using_config(seg_cfg,
                                                   track_cfg,
                                                   trace_cfg,
                                                   project_cfg,
                                                   DEBUG=DEBUG)
