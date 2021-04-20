"""
The top level function for initializing a stack of DLC projects
"""

from pathlib import Path
# main function
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd
from DLC_for_WBFM.utils.preprocessing.DLC_utils import train_all_dlc_from_config
# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment()
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    project_dir = Path(project_path).parent

    with safe_cd(project_dir):
        tracking_fname = Path(project_cfg['subfolder_configs']['tracking'])
        tracking_cfg = dict(load_config(tracking_fname))


@ex.automain
def initialize_dlc_stack(_config, _run):
    sacred.commands.print_config(_run)

    this_config = _config['tracking_cfg'].copy()
    with safe_cd(_config['project_dir']):
        train_all_dlc_from_config(this_config)
