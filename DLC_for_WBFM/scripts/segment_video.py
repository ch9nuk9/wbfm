"""
The top level functions for segmenting a full (WBFM) recording.

To be used with Niklas' Stardist-based segmentation package
"""


import os
from pathlib import Path
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, edit_config, synchronize_segment_config
from segmentation.util.utils_pipeline import segment_video_using_config_2d
# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Add sub-config file as found in main project config
# TODO: put in function?
# tmp_cfg = load_config('project_config.yaml')
# segment_cfg = load_config(tmp_cfg['subfolder_configs']['segmentation'])

# Initialize sacred experiment
ex = Experiment()
# TODO: script must be run from project folder for now
# ex.add_config('project_config.yaml')
ex.add_config(project_path=None)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    segment_fname = Path(project_cfg['subfolder_configs']['segmentation'])
    # print(project_path, segment_fname)
    project_dir = Path(project_path).parent
    segment_fname = Path(project_dir).joinpath(segment_fname)
    segment_cfg = dict(load_config(segment_fname))

    # Sync filename in segmentation config from project_cfg
    segment_cfg = synchronize_segment_config(project_path, segment_cfg)
    edit_config(segment_fname, segment_cfg)


@ex.automain
def segment2d(_config, _run):
    sacred.commands.print_config(_run)

    segment_video_using_config_2d(_config['segment_cfg'])
