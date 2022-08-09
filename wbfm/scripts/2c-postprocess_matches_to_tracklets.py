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
from wbfm.pipeline.tracklets import postprocess_matches_to_tracklets_using_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'  # Capture stdout
import cgitb
cgitb.enable(format='text')

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_2b.log')

    train_cfg = cfg.get_training_config()
    segmentation_config = cfg.get_segmentation_config()

    if not DEBUG:
        using_monkeypatch()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    project_config = _config['cfg']
    train_cfg = _config['train_cfg']
    segmentation_config = _config['segmentation_config']

    postprocess_matches_to_tracklets_using_config(
        project_config,
        segmentation_config,
        train_cfg,
        DEBUG=DEBUG
    )
