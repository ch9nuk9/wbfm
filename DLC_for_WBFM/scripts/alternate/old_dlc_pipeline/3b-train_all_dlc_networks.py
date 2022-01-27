"""
The top level function for initializing a stack of DLC projects
"""

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
# main function
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch

from DLC_for_WBFM.utils.traces.dlc_pipeline import train_all_dlc_from_config
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)

    tracking_cfg = cfg.get_tracking_config()

    if not DEBUG:
        using_monkeypatch
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def initialize_dlc_stack(_config, _run):
    sacred.commands.print_config(_run)

    tracking_cfg = _config['tracking_cfg']
    train_all_dlc_from_config(tracking_cfg)
