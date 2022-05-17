"""
The top level function for producing training data via feature-based tracking
"""
import logging
import os
from datetime import date

# Experiment tracking
import sacred
from sacred import Experiment
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch
from DLC_for_WBFM.utils.projects.utils_project_status import check_all_needed_data_for_step
from DLC_for_WBFM.pipeline.tracklets import build_frame_objects_using_config
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig, update_path_to_segmentation_in_config

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'  # Capture stdout
import cgitb
cgitb.enable(format='text')

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, only_calculate_desynced=False, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    check_all_needed_data_for_step(cfg, 2)

    train_cfg = update_path_to_segmentation_in_config(cfg)
    train_cfg.update_self_on_disk()

    log_dir = cfg.get_log_dir()
    log_fname = os.path.join(log_dir, '2-training_data_warnings.log')
    logging.basicConfig(filename=log_fname, level=logging.DEBUG)
    logging.warning(f'Starting run at: {date.today().strftime("%Y/%m/%d %H:%M:%S")}')
    if not DEBUG:
        using_monkeypatch()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    project_config = _config['cfg']
    only_calculate_desynced = _config['only_calculate_desynced']

    train_cfg = _config['train_cfg']

    build_frame_objects_using_config(
        project_config,
        train_cfg,
        only_calculate_desynced=only_calculate_desynced,
        DEBUG=DEBUG
    )
