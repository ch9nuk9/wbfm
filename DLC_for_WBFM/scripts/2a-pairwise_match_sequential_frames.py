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
from DLC_for_WBFM.pipeline.tracklets import build_frames_and_adjacent_matches_using_config
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig, update_path_to_segmentation_in_config

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'  # Capture stdout
import cgitb
cgitb.enable(format='text')

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, use_superglue=True, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_2a.log')
    check_all_needed_data_for_step(project_path, 2)

    train_cfg = update_path_to_segmentation_in_config(cfg)
    train_cfg.update_self_on_disk()

    if not DEBUG:
        using_monkeypatch()


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    use_superglue = _config['use_superglue']
    project_config = _config['cfg']
    train_cfg = _config['train_cfg']

    build_frames_and_adjacent_matches_using_config(
        project_config,
        train_cfg,
        use_superglue=use_superglue,
        DEBUG=DEBUG
    )
