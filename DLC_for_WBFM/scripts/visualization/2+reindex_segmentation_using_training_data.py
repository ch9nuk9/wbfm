"""
The top level function for producing dlc tracks in 3d
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
# Initialize sacred experiment
from DLC_for_WBFM.utils.visualization.utils_segmentation import reindex_segmentation_only_training_data
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config

ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    cfg = modular_project_config(project_path)
    project_dir = cfg.project_dir

    segment_cfg = cfg.get_segmentation_config()
    tracking_cfg = cfg.get_tracking_config()


@ex.automain
def make_dlc_labeled_videos(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']

    # For manual correction and visualization
    reindex_segmentation_only_training_data(
        _config['cfg'],
        _config['segment_cfg'],
        _config['tracking_cfg'],
        DEBUG = DEBUG
    )
