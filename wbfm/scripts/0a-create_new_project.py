import os

import sacred
from sacred import Experiment
from sacred import SETTINGS
import cgitb
from wbfm.utils.projects.utils_filenames import get_location_of_new_project_defaults

cgitb.enable(format='text')
from wbfm.pipeline.project_initialization import build_project_structure_from_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment(save_git_info=False)
path = get_location_of_new_project_defaults()
ex.add_config(os.path.join(path, 'project_config.yaml'))


@ex.config
def cfg(red_bigtiff_fname, green_bigtiff_fname, parent_data_folder):
    pass
    # if not DEBUG:
    #     assert osp.exists(red_bigtiff_fname)
    #     assert osp.exists(green_bigtiff_fname)


@ex.automain
def main(_config, _run, _log):
    sacred.commands.print_config(_run)

    build_project_structure_from_config(_config, _log)
