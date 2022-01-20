"""
The top level function for producing training data via feature-based tracking
"""
import logging
import os
from datetime import date

# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import alt_save_all_tracklets_as_dlc_format
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'  # Capture stdout

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, allow_raw_artifact_reuse=False, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)

    train_cfg = cfg.get_training_config()

    log_dir = cfg.get_log_dir()
    log_fname = os.path.join(log_dir, '2-training_data_warnings.log')
    logging.basicConfig(filename=log_fname, level=logging.DEBUG)
    logging.warning(f'Starting run at: {date.today().strftime("%Y/%m/%d %H:%M:%S")}')
    if not DEBUG:
        using_monkeypatch()
        ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    project_config = _config['cfg']
    train_cfg = _config['train_cfg']

    # For later extending
    with safe_cd(project_config.project_dir):
        alt_save_all_tracklets_as_dlc_format(train_cfg)
