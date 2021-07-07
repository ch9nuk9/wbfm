"""
The top level functions for segmenting a full (WBFM) recording.

To be used with Niklas' Stardist-based segmentation package
"""


import os
from pathlib import Path
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config, synchronize_segment_config, safe_cd
from segmentation.util.utils_pipeline import segment_video_using_config_2d, segment_video_using_config_3d
# Experiment tracking
import sacred
from sacred import Experiment

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, continue_from_frame=None)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    segment_fname = Path(project_cfg['subfolder_configs']['segmentation'])
    project_dir = Path(project_path).parent
    segment_fname = Path(project_dir).joinpath(segment_fname)
    segment_cfg = dict(load_config(segment_fname))

    # Sync filename in segmentation config from project_cfg
    segment_cfg = synchronize_segment_config(project_path, segment_cfg)
    edit_config(segment_fname, segment_cfg)


@ex.automain
def segment2d(_config, _run):
    sacred.commands.print_config(_run)

    # For windows workstation
    os.environ['NUMEXPR_MAX_THREADS'] = '56'

    this_config = _config['segment_cfg'].copy()
    this_config['dataset_params'] = _config['project_cfg']['dataset_params'].copy()

    with safe_cd(_config['project_dir']):
        mode = this_config['segmentation_type']
        if mode == "3d":
            segment_video_using_config_3d(this_config, _config['continue_from_frame'])
        elif mode == "2d":
            segment_video_using_config_2d(this_config)
        else:
            raise ValueError(f"Unknown segmentation_type; expected '2d' or '3d' instead of {mode}")