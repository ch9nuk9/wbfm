# main function
from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch
from segmentation.util.utils_metadata import recalculate_metadata_from_config
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig

from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, out_fname=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    segment_cfg = cfg.get_segmentation_config()
    train_cfg = cfg.get_training_config()

    # Modify the config files so that we process the training data instead
    segment_cfg.config['output_masks'] = train_cfg.config['reindexed_masks']
    segment_cfg.config['output_metadata'] = train_cfg.config['reindexed_metadata']
    # TODO: check that start frame indexes properly
    cfg.config['dataset_params']['num_frames'] = train_cfg.config['training_data_3d']['num_training_frames']

    if not DEBUG:
        using_monkeypatch()
        log_dir = cfg.get_log_dir()
        ex.observers.append(TinyDbObserver(log_dir))

@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    segment_cfg = _config['segment_cfg']
    project_cfg = _config['cfg']

    with safe_cd(_config['project_dir']):
        recalculate_metadata_from_config(segment_cfg, project_cfg, _config['DEBUG'])
