"""
The top level function for initializing a stack of DLC projects
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch

# main function
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig
from DLC_for_WBFM.utils.preprocessing.DLC_utils import update_all_pose_configs

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, update_key=None, update_val=None, DEBUG=False)


@ex.config
def cfg(project_path, update_key, update_val, DEBUG):
    cfg = ModularProjectConfig(project_path)

    tracking_cfg = cfg.get_tracking_config()
    updates = {update_key: update_val}

    if not DEBUG:
        using_monkeypatch()
        log_dir = cfg.get_log_dir()
        ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def update_all_pose_configs_in_project(_config, _run):
    sacred.commands.print_config(_run)

    update_all_pose_configs(_config['tracking_cfg'], updates=_config['updates'])
