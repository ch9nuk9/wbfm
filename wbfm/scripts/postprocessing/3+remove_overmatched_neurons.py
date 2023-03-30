"""
"""

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from sacred.observers import TinyDbObserver

from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.general.postprocessing.combine_tracklets_and_spatial_tracks import remove_overmatched_tracks_using_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    tracking_cfg = cfg.get_tracking_config()

    if not DEBUG:
        using_monkeypatch()
        log_dir = cfg.get_log_dir()
        ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def combine_tracks(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    track_cfg = _config['tracking_cfg']

    remove_overmatched_tracks_using_config(track_cfg)
