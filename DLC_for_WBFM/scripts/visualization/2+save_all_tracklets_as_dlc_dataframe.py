"""
The top level function for producing dlc tracks in 3d
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
# Initialize sacred experiment
from DLC_for_WBFM.utils.training_data.tracklet_to_DLC import alt_save_all_tracklets_as_dlc_format

ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, min_length=10)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir
    train_cfg = cfg.get_training_config()


@ex.automain
def save_training_data(_config, _run):
    sacred.commands.print_config(_run)

    train_cfg = _config['train_cfg']

    with safe_cd(_config['project_dir']):
        alt_save_all_tracklets_as_dlc_format(train_cfg)
