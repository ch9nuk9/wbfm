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
def cfg(red_fname, green_fname, parent_data_folder):
    pass


@ex.automain
def main(_config, _run, _log):
    """
    Example:
    python 0a-create_new_project.py with
        parent_data_folder='/scratch/neurobiology/zimmer/ulises/wbfm/20220824/data/ZIM2319_worm2'
        project_dir='/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms'
        task_name=gfp
        experimenter=C
    """
    sacred.commands.print_config(_run)

    build_project_structure_from_config(_config, _log)
