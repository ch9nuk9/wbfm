"""
The top level functions for segmenting a full (WBFM) recording.

To be used with Niklas' Stardist-based segmentation package
"""

import os
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch
from segmentation.util.utils_pipeline import segment_video_using_config_2d, segment_video_using_config_3d

from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig, synchronize_segment_config
import cgitb

from DLC_for_WBFM.utils.projects.utils_project_status import check_all_needed_data_for_step

cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, continue_from_frame=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    check_all_needed_data_for_step(project_path, 1)
    project_dir = cfg.project_dir

    segment_cfg = cfg.get_segmentation_config()
    segment_cfg.config = synchronize_segment_config(project_path, segment_cfg.config)
    segment_cfg.update_on_disk()

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def segment2d(_config, _run):
    sacred.commands.print_config(_run)

    # For windows workstation
    os.environ['NUMEXPR_MAX_THREADS'] = '56'

    segment_cfg = _config['segment_cfg']
    project_cfg = _config['cfg']

    if _config['DEBUG']:
        project_cfg.config['dataset_params']['num_frames'] = 3

    mode = segment_cfg.config['segmentation_type']
    with safe_cd(project_cfg.project_dir):
        if mode == "3d":
            segment_video_using_config_3d(segment_cfg, project_cfg, _config['continue_from_frame'], DEBUG=_config['DEBUG'])
        elif mode == "2d":
            segment_video_using_config_2d(segment_cfg, project_cfg, _config['continue_from_frame'], DEBUG=_config['DEBUG'])
        else:
            raise ValueError(f"Unknown segmentation_type; expected '2d' or '3d' instead of {mode}")
