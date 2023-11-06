
# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.utils_redo_steps import remap_tracklets_to_new_segmentation_using_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, new_segmentation_suffix=None, path_to_new_segmentation=None, path_to_new_metadata=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    project_path = _config['project_path']
    path_to_new_segmentation = _config['path_to_new_segmentation']
    path_to_new_metadata = _config['path_to_new_metadata']
    new_segmentation_suffix = _config['new_segmentation_suffix']
    DEBUG = _config['DEBUG']

    remap_tracklets_to_new_segmentation_using_config(project_path, new_segmentation_suffix,
                                                     path_to_new_segmentation, path_to_new_metadata,
                                                     DEBUG)
