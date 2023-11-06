# main function
from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from segmentation.util.utils_pipeline import resplit_masks_in_z_from_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

from wbfm.utils.projects.utils_project import safe_cd
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
    project_dir = cfg.project_dir

    segment_cfg = cfg.get_segmentation_config()

    segment_cfg.config['postprocessing_params']['already_stitched'] = True

    if not DEBUG:
        using_monkeypatch()
    #     log_dir = cfg.get_log_dir()
    #     ex.observers.append(TinyDbObserver(log_dir))

@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    segment_cfg = _config['segment_cfg']
    project_cfg = _config['cfg']

    with safe_cd(_config['project_dir']):
        resplit_masks_in_z_from_config(segment_cfg, project_cfg,
                                       continue_from_frame=_config['continue_from_frame'],
                                       DEBUG=_config['DEBUG'])
