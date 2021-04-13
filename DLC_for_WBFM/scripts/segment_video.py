"""
The top level functions for segmenting a full (WBFM) recording.

To be used with Niklas' Stardist-based segmentation package
"""


import os
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config
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

@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config('project_config.yaml')
    segment_fname = project_cfg['subfolder_configs']['segmentation']
    segment_cfg = dict(load_config(segment_fname))


@ex.automain
def segment2d(_config, _run):
    sacred.commands.print_config(_run)

    segment_video_using_config_2d(_config['segment_cfg'])
