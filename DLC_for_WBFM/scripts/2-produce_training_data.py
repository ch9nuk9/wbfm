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

from DLC_for_WBFM.utils.pipeline.tracklet_pipeline import partial_track_video_using_config
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config, update_path_to_segmentation_in_config

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = modular_project_config(project_path)
    project_dir = cfg.project_dir

    train_cfg = update_path_to_segmentation_in_config(cfg)
    train_cfg.update_on_disk()

    log_dir = cfg.get_log_dir()
    log_fname = os.path.join(log_dir, '2-training_data_warnings.log')
    logging.basicConfig(filename=log_fname, level=logging.DEBUG)
    logging.warning(f'Starting run at: {date.today().strftime("%Y/%m/%d %H:%M:%S")}')
    if not DEBUG:
        ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    training_config = _config['train_cfg']
    project_config = _config['cfg']

    partial_track_video_using_config(project_config, training_config, DEBUG=DEBUG)
