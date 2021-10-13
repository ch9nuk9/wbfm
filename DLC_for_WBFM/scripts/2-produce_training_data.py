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
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import save_training_data_as_dlc_format, \
    save_all_tracklets_as_dlc_format
from DLC_for_WBFM.utils.visualization.utils_segmentation import reindex_segmentation_only_training_data
from DLC_for_WBFM.utils.pipeline.tracklet_pipeline import partial_track_video_using_config
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config, update_path_to_segmentation_in_config

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys' # Capture stdout

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

    tracking_cfg = cfg.get_tracking_config()
    segment_cfg = cfg.get_segmentation_config()

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

    DEBUG = _config['DEBUG']
    project_config = _config['cfg']

    tracking_cfg = _config['tracking_cfg']
    segment_cfg = _config['segment_cfg']
    train_cfg = _config['train_cfg']

    partial_track_video_using_config(
        project_config,
        train_cfg,
        DEBUG=DEBUG
    )

    # For manual correction
    reindex_segmentation_only_training_data(
        project_config,
        segment_cfg,
        tracking_cfg,
        DEBUG=DEBUG
    )

    # For later analysis, i.e. don't use the raw dataframes directly
    save_training_data_as_dlc_format(tracking_cfg,
                                     train_cfg, DEBUG=DEBUG)

    # For later extending
    min_length = train_cfg.config['postprocessing_params']['min_length_to_save']
    save_all_tracklets_as_dlc_format(None, min_length=min_length)
