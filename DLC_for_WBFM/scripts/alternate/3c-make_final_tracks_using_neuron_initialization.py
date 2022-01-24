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
from DLC_for_WBFM.utils.pipeline.long_range_matching import global_track_matches_from_config
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
import cgitb
cgitb.enable(format='text')

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, use_imputed_df=False, start_from_manual_matches=True, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def combine_tracks(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    global_track_matches_from_config(_config['project_path'], DEBUG=DEBUG)
