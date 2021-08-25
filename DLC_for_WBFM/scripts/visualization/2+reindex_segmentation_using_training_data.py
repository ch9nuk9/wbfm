"""
The top level function for producing dlc tracks in 3d
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

from DLC_for_WBFM.utils.pipeline.dlc_pipeline import make_all_dlc_labeled_videos
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
# Initialize sacred experiment
from DLC_for_WBFM.utils.visualization.utils_segmentation import reindex_segmentation_only_training_data

ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    project_dir = Path(project_path).parent

    with safe_cd(project_dir):
        track_fname = Path(project_cfg['subfolder_configs']['tracking'])
        track_cfg = dict(load_config(track_fname))

        segment_fname = Path(project_cfg['subfolder_configs']['segmentation'])
        segment_cfg = dict(load_config(segment_fname))


@ex.automain
def make_dlc_labeled_videos(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    this_config = _config.copy()
    this_config['dataset_params'] = _config['project_cfg']['dataset_params'].copy()

    with safe_cd(_config['project_dir']):
        reindex_segmentation_only_training_data(this_config, DEBUG=DEBUG)
