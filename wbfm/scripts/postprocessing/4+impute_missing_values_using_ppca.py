"""
The top level function for producing dlc tracks in 3d

EXPERIMENTAL
"""

from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment

# main function
from wbfm.utils.general.postprocessing.utils_imputation import impute_missing_values_using_config
from wbfm.utils.projects.utils_project import safe_cd
from wbfm.utils.external.utils_yaml import load_config

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, filter_mode='arima', DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    project_cfg = load_config(project_path)
    project_dir = Path(project_path).parent

    with safe_cd(project_dir):
        track_fname = Path(project_cfg['subfolder_configs']['tracking'])
        track_cfg = dict(load_config(track_fname))


@ex.automain
def make_dlc_labeled_videos(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    this_config = _config['track_cfg'].copy()
    this_config['project_dir'] = _config['project_dir']

    with safe_cd(_config['project_dir']):
        impute_missing_values_using_config(this_config, DEBUG=DEBUG)
