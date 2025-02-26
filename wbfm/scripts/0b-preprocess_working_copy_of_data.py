"""
"""

import sacred
from sacred import Experiment
from sacred import SETTINGS
from wbfm.utils.external.monkeypatch_json import using_monkeypatch

from wbfm.pipeline.project_initialization import preprocess_fluorescence_data
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, to_zip_zarr_using_7z=True, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_0b.log')

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    cfg: ModularProjectConfig = _config['cfg']
    cfg.config['project_path'] = _config['project_path']
    to_zip_zarr_using_7z = _config['to_zip_zarr_using_7z']
    DEBUG = _config['DEBUG']

    preprocess_fluorescence_data(cfg, to_zip_zarr_using_7z, DEBUG)
