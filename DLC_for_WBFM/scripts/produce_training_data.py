"""
The top level function for producing training data via feature-based tracking
"""

from pathlib import Path
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config, safe_cd, synchronize_train_config
from DLC_for_WBFM.utils.pipeline.tracklet_pipeline import partial_track_video_using_config
# Experiment tracking
import sacred
from sacred import Experiment

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    project_dir = Path(project_path).parent

    with safe_cd(project_dir):
        train_fname = Path(project_cfg['subfolder_configs']['training_data'])
        train_cfg = dict(load_config(train_fname))

        train_cfg = synchronize_train_config(Path(project_path).name, train_cfg)
        edit_config(train_fname, train_cfg)


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    vid_fname = _config['project_cfg']['red_bigtiff_fname']
    this_config = _config['train_cfg'].copy()
    this_config['dataset_params'] = _config['project_cfg']['dataset_params'].copy()

    with safe_cd(_config['project_dir']):
        partial_track_video_using_config(vid_fname, this_config, DEBUG=DEBUG)
