import sacred
from sacred import Experiment
from sacred import SETTINGS
import cgitb
cgitb.enable(format='text')
from DLC_for_WBFM.pipeline.project_initialization import build_project_structure_from_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment()
ex.add_config('new_project_defaults/project_config.yaml')


@ex.config
def cfg(red_bigtiff_fname, green_bigtiff_fname):
    pass
    # if not DEBUG:
    #     assert osp.exists(red_bigtiff_fname)
    #     assert osp.exists(green_bigtiff_fname)


@ex.automain
def main(_config, _run, _log):
    sacred.commands.print_config(_run)

    build_project_structure_from_config(_config, _log)
