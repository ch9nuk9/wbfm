"""
"""

# Experiment tracking
import sacred

from sacred import Experiment
from sacred import SETTINGS
from wbfm.utils.external.monkeypatch_json import using_monkeypatch

from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.nwb.utils_nwb_unpack import unpack_nwb_to_project_structure
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)

    if not DEBUG:
        using_monkeypatch()


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    unpack_nwb_to_project_structure(_config['cfg'])

    print("Finished.")

