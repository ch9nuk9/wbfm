"""
The top level function for producing training data via feature-based tracking
"""

import os
from pathlib import Path
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config, safe_cd
from segmentation.util.utils_pipeline import segment_video_using_config_2d
# Experiment tracking
import sacred
from sacred import Experiment

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    train_fname = Path(project_cfg['subfolder_configs']['training_data'])
    project_dir = Path(project_path).parent
    train_fname = Path(project_dir).joinpath(train_fname)
    train_cfg = dict(load_config(train_fname))

    # TODO: Sync configs
    train_cfg = synchronize_train_config(project_path, train_cfg)
    edit_config(train_fname, train_cfg)


@ex.automain
def produce_training_data(_config, _run):
    sacred.commands.print_config(_run)

    vid_fname = _config['red_bigtiff_fname']
    opt = {}
    opt['scorer'] = _config['experimenter']

    with safe_cd(_config['project_dir']):
        partial_track_video_using_config(vid_fname, _config['train_cfg'], **opt)
