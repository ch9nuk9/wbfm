"""
The top level function for producing training data via feature-based tracking
"""
import logging
import os
from datetime import date

# Experiment tracking
import sacred
from sacred import Experiment
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.pipeline.tracklets import build_frames_and_adjacent_matches_using_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig, update_path_to_segmentation_in_config

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'  # Capture stdout
import cgitb
cgitb.enable(format='text')

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, use_superglue=True, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_2a.log')
    check_all_needed_data_for_step(cfg, 2)

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
