"""
The top level functions for segmenting a full (WBFM) recording.

To be used with Niklas' Stardist-based segmentation package
"""

import os
from wbfm.utils.projects.utils_project import safe_cd
# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from segmentation.util.utils_pipeline import segment_video_using_config_2d, segment_video_using_config_3d

from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, continue_from_frame=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_1.log')
    check_all_needed_data_for_step(cfg, 1)
    project_dir = cfg.project_dir

    segment_cfg = cfg.get_segmentation_config()
    preprocessing_cfg = cfg.get_preprocessing_config()

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def segment_video(_config, _run):
    sacred.commands.print_config(_run)

    # For windows workstation
    os.environ['NUMEXPR_MAX_THREADS'] = '56'
    # Set environment variables to (try to) deal with rare blosc decompression errors
    os.environ["BLOSC_NOLOCK"] = "1"
    os.environ["BLOSC_NTHREADS"] = "1"
    # Tensorflow has memory flushing problems, so disallow gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    segment_cfg = _config['segment_cfg']
    preprocessing_cfg = _config['preprocessing_cfg']
    project_cfg = _config['cfg']
    if _config['DEBUG']:
        project_cfg.config['dataset_params']['num_frames'] = 3

    mode = segment_cfg.config['segmentation_type']
    opt = dict(preprocessing_cfg=preprocessing_cfg, segment_cfg=segment_cfg, project_cfg=project_cfg,
               continue_from_frame=_config['continue_from_frame'], DEBUG=_config['DEBUG'])
    with safe_cd(project_cfg.project_dir):
        if mode == "3d":
            segment_video_using_config_3d(**opt)
        elif mode == "2d":
            segment_video_using_config_2d(**opt)
        else:
            raise ValueError(f"Unknown segmentation_type; expected '2d' or '3d' instead of {mode}")
