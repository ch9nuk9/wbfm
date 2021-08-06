"""
The top level function for producing training data via feature-based tracking
"""

from pathlib import Path
# main function
from sacred.observers import TinyDbObserver
import DLC_for_WBFM.utils.projects.monkeypatch_json

from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config, update_path_to_segmentation_in_config
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config, safe_cd
from DLC_for_WBFM.utils.pipeline.tracklet_pipeline import partial_track_video_using_config
# Experiment tracking
import sacred
from sacred import Experiment

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

    if not DEBUG:
        log_dir = str(Path(project_dir).joinpath('log'))
        ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    # vid_fname = _config['project_cfg']['red_bigtiff_fname']
    # vid_fname = _config['cfg'].config['preprocessed_red']
    training_config = _config['train_cfg'].copy()
    project_config = _config['cfg'].copy()
    # training_config['dataset_params'] = _config['project_cfg']['dataset_params'].copy()

    with safe_cd(_config['project_dir']):
        partial_track_video_using_config(project_config, training_config, DEBUG=DEBUG)
