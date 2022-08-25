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
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

from sacred import SETTINGS

from wbfm.utils.tracklets.training_data_from_tracklets import save_training_data_as_dlc_format

SETTINGS.CAPTURE_MODE = 'sys' # Capture stdout

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, allow_raw_artifact_reuse=False, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir
    train_cfg = cfg.get_training_config()

    log_dir = cfg.get_log_dir()
    # log_fname = os.path.join(log_dir, '2-training_data_warnings.log')
    # logging.basicConfig(filename=log_fname, level=logging.DEBUG)
    # logging.warning(f'Starting run at: {date.today().strftime("%Y/%m/%d %H:%M:%S")}')
    if not DEBUG:
        using_monkeypatch()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']

    train_cfg = _config['train_cfg']

    # For later analysis, i.e. don't use the raw dataframes directly
    save_training_data_as_dlc_format(train_cfg, DEBUG=DEBUG)
