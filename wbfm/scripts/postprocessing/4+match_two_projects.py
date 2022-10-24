"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

# Experiment tracking
import sacred

from wbfm.pipeline.tracking import match_two_projects_using_superglue_using_config
from wbfm.utils.nn_utils.track_using_barlow import track_using_barlow_from_config
from sacred import Experiment
from sacred import SETTINGS
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.utils.projects.project_config_classes import ModularProjectConfig

import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path_base=None, project_path_target=None, DEBUG=False)


@ex.config
def cfg(project_path_base, project_path_target, DEBUG):
    # Manually load yaml files
    cfg_base = ModularProjectConfig(project_path_base)
    check_all_needed_data_for_step(cfg_base, 4, training_data_required=False)

    cfg_target = ModularProjectConfig(project_path_base)
    cfg_target.setup_logger('step_4+.log')
    check_all_needed_data_for_step(cfg_target, 4, training_data_required=False)

    if not DEBUG:
        using_monkeypatch()


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    project_path_base = _config['cfg_base']
    project_path_target = _config['cfg_target']

    match_two_projects_using_superglue_using_config(project_path_base, project_path_target, DEBUG)
