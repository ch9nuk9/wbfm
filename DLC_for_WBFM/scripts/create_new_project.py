
from DLC_for_WBFM.utils.projects.utils_projects import build_project_structure
import os.path as osp

import sacred
from sacred import Experiment


ex = Experiment()
ex.add_config('new_project_defaults/project_config.yaml')


@ex.config
def cfg(path, red_bigtiff_fname, green_bigtiff_fname):
    # assert osp.exists(red_bigtiff_fname)
    # assert osp.exists(green_bigtiff_fname)
    pass

@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    build_project_structure(_config, path)
