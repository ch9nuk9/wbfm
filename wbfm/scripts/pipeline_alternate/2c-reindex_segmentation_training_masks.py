"""
The top level function for producing training data via feature-based tracking
"""
import logging
import os
from datetime import date
import cgitb

from wbfm.utils.tracklets.training_data_from_tracklets import modify_config_files_for_training_data

cgitb.enable(format='text')

# Experiment tracking
import sacred

from wbfm.utils.traces.traces_pipeline import extract_traces_of_training_data_from_config
from wbfm.utils.projects.utils_project import safe_cd
from sacred import Experiment
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.visualization.utils_segmentation import reindex_segmentation_only_training_data
from wbfm.utils.segmentation.util.utils_metadata import recalculate_metadata_from_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

from sacred import SETTINGS
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

    training_cfg = cfg.get_training_config()
    segment_cfg = cfg.get_segmentation_config()
    preprocessing_cfg = cfg.get_preprocessing_config()

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

    training_cfg = _config['training_cfg']
    segment_cfg = _config['segment_cfg']
    preprocessing_cfg = _config['preprocessing_cfg']

    # For manual correction or improvement of later algorithms
    reindex_segmentation_only_training_data(
        project_config,
        segment_cfg,
        training_cfg,
        DEBUG=DEBUG
    )

    modify_config_files_for_training_data(project_config, segment_cfg, training_cfg)

    with safe_cd(project_config.project_dir):
        recalculate_metadata_from_config(preprocessing_cfg, segment_cfg, project_config, name_mode='tracklet', DEBUG=DEBUG)

        extract_traces_of_training_data_from_config(project_config, training_cfg)
        # save_training_data_as_dlc_format(training_cfg, segment_cfg, DEBUG=DEBUG)
