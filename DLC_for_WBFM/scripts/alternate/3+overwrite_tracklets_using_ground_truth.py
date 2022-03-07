"""
The top level function for producing dlc tracks in 3d
"""

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
# main function
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch

from DLC_for_WBFM.utils.traces.dlc_pipeline import make_3d_tracks_from_stack
from DLC_for_WBFM.utils.general.postprocessing.combine_tracklets_and_DLC_tracks import \
    match_dlc_and_tracklet_coverings_from_config
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.tracklets.tracklet_pipeline import overwrite_tracklets_using_ground_truth

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

    tracking_cfg = cfg.get_tracking_config()
    training_cfg = cfg.get_training_config()

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
