import os

import sacred
from sacred import Experiment
from sacred import SETTINGS
import cgitb
from wbfm.utils.general.utils_filenames import get_location_of_new_project_defaults

cgitb.enable(format='text')
from wbfm.pipeline.project_initialization import build_project_structure_from_config, build_project_structure_from_nwb_file

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment(save_git_info=False)
ex.add_config(project_dir=None, nwb_file=None, copy_nwb_file=True)

@ex.config
def cfg(project_dir, nwb_file, copy_nwb_file):
    pass


@ex.automain
def main(_config, _run, _log):
    """
    Example:
    python 0a-create_new_project_from_nwb.py with
        nwb_file='/path/to/your/data.nwb'
        project_dir='/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms'
        task_name=gfp
        experimenter=C
        # OPTIONAL:
        neuropal_path='/scratch/neurobiology/zimmer/ulises/wbfm/20220824/data/ZIM2319_worm2/20220824_ZIM2319_worm2_gfp_neuropal.h5'

    See also wbfm/scripts/0a-create_new_project.py
    """
    sacred.commands.print_config(_run)

    build_project_structure_from_nwb_file(_config, _config['nwb_file'], _config['copy_nwb_file'])
