"""
The top level function for producing dlc tracks in 3d
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from sacred.observers import TinyDbObserver

from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch
from DLC_for_WBFM.utils.postprocessing.combine_tracklets_and_DLC_tracks import \
    combine_all_dlc_and_tracklet_coverings_from_config
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, use_imputed_df=False, start_from_manual_matches=False, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    tracking_cfg = cfg.get_tracking_config()
    training_cfg = cfg.get_training_config()

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def combine_tracks(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    track_cfg = _config['tracking_cfg']
    training_cfg = _config['training_cfg']

    combine_all_dlc_and_tracklet_coverings_from_config(track_cfg, training_cfg, _config['cfg'],
                                                       use_imputed_df=_config['use_imputed_df'],
                                                       start_from_manual_matches=_config['start_from_manual_matches'],
                                                       DEBUG=DEBUG)
