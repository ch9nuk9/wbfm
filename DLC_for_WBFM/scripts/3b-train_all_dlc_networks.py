"""
The top level function for initializing a stack of DLC projects
"""

from pathlib import Path
# main function
from sacred.observers import TinyDbObserver
import DLC_for_WBFM.utils.projects.monkeypatch_json

from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
from DLC_for_WBFM.utils.pipeline.dlc_pipeline import train_all_dlc_from_config
# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    cfg = modular_project_config(project_path)

    tracking_cfg = cfg.get_tracking_config()

    log_dir = cfg.get_log_dir()
    ex.observers.append(TinyDbObserver(log_dir))

@ex.automain
def initialize_dlc_stack(_config, _run):
    sacred.commands.print_config(_run)

    tracking_cfg = _config['tracking_cfg']
    train_all_dlc_from_config(tracking_cfg)
