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
from DLC_for_WBFM.utils.feature_detection.custom_errors import AnalysisOutOfOrderError
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import save_training_data_as_dlc_format, \
    save_all_tracklets_as_dlc_format
from DLC_for_WBFM.utils.visualization.utils_segmentation import reindex_segmentation_only_training_data
from DLC_for_WBFM.utils.pipeline.tracklet_pipeline import partial_track_video_using_config
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig, update_path_to_segmentation_in_config

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys' # Capture stdout

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, allow_raw_artifact_reuse=False, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    tracking_cfg = cfg.get_tracking_config()
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

    DEBUG = _config['DEBUG']

    tracking_cfg = _config['tracking_cfg']
    train_cfg = _config['train_cfg']

    # For later analysis, i.e. don't use the raw dataframes directly
    save_training_data_as_dlc_format(tracking_cfg,
                                     train_cfg, DEBUG=DEBUG)
