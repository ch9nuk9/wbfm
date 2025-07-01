# main function
from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.segmentation.util.utils_metadata import recalculate_metadata_from_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.external.utils_yaml import load_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, out_fname=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    project_config = ModularProjectConfig(project_path)

@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    recalculate_metadata_from_config(_config['project_config'], name_mode='neuron', DEBUG=_config['DEBUG'])
