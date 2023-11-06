"""
The top level function for producing dlc tracks in 3d
"""

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
# main function
from wbfm.utils.external.monkeypatch_json import using_monkeypatch

from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.tracklets.tracklet_pipeline import overwrite_tracklets_using_ground_truth

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def make_full_tracks(_config, _run):
    sacred.commands.print_config(_run)

    cfg = _config['cfg']
    DEBUG = _config['DEBUG']

    overwrite_tracklets_using_ground_truth(cfg, DEBUG=DEBUG)
