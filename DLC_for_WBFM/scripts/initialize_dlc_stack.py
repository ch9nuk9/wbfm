"""
The top level function for initializing a stack of DLC projects
"""

import os
from pathlib import Path
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config, safe_cd
from DLC_for_WBFM.utils.preprocessing.DLC_utils import create_dlc_training_from_tracklets
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
        tracking_fname = Path(project_cfg['subfolder_configs']['tracking'])
        tracking_cfg = dict(load_config(tracking_fname))


@ex.automain
def initialize_dlc_stack(_config, _run):
    sacred.commands.print_config(_run)

    vid_fname = _config['red_bigtiff_fname']
    this_config = _config['tracking_cfg']
    this_config['dataset_params'] = _config['project_cfg']['dataset_params']

    opt = {}
    opt['scorer'] = _config['project_cfg']['experimenter']
    opt['task_name'] = _config['project_cfg']['experimenter']
    opt['DEBUG'] = _config['DEBUG']

    with safe_cd(_config['project_dir']):
        create_dlc_training_from_tracklets(vid_fname, this_config, **opt)
