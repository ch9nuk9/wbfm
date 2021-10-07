"""
"""

# main function
import os

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch
from DLC_for_WBFM.utils.preprocessing.bounding_boxes import calculate_bounding_boxes_from_fnames
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment()
ex.add_config(project_path=None,
              DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = modular_project_config(project_path)
    bounding_box_fname = os.path.join(cfg.project_dir, '1-segmentation', 'bounding_boxes.pickle')

    segment_cfg = cfg.get_segmentation_config()

    if not DEBUG:
        using_monkeypatch()
        log_dir = cfg.get_log_dir()
        ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    video_fname = _config['cfg'].config['preprocessed_red']
    bbox_fname = _config['bounding_box_fname']
    calculate_bounding_boxes_from_fnames(video_fname, bbox_fname)

    segment_cfg = _config['segment_cfg']
    segment_cfg.config['bbox_fname'] = bbox_fname
    segment_cfg.update_on_disk()
