"""
The top level function for producing dlc tracks in 3d
"""

# Experiment tracking
import sacred
from wbfm.utils.general.postprocessing.utils_imputation import impute_tracks_from_config
from sacred import Experiment

# main function
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    tracking_cfg = cfg.get_tracking_config()

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def combine_tracks(_config, _run):
    sacred.commands.print_config(_run)

    tracking_cfg = _config['tracking_cfg']
    impute_tracks_from_config(tracking_cfg)
