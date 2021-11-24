# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig

from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.nn_utils.fdnc_predict import track_using_fdnc_from_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
ex.add_config(project_path=None, out_fname=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    tracks_cfg = cfg.get_tracking_config()

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    tracks_cfg = _config['tracks_cfg']
    project_cfg = _config['cfg']
    project_dir = _config['project_dir']

    with safe_cd(project_dir):
        track_using_fdnc_from_config(project_cfg, tracks_cfg, _config['DEBUG'])

        # Necessary postprocessing step
        # combine_all_dlc_and_tracklet_coverings_from_config(tracks_cfg, training_cfg, project_dir, DEBUG=_config['DEBUG'])
