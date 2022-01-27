"""
The top level function for producing dlc tracks in 3d
"""

# Experiment tracking
import sacred
# Initialize sacred experiment
from sacred import Experiment

from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
# main function
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
from DLC_for_WBFM.utils.tracklets.tracklet_to_DLC import save_training_data_as_dlc_format

ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir
    training_cfg = cfg.get_training_config()


@ex.automain
def save_training_data(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']

    with safe_cd(_config['project_dir']):
        save_training_data_as_dlc_format(_config['training_cfg'], DEBUG=DEBUG)
