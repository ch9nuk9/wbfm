"""
"""

# main function
import os
from pathlib import Path

from DLC_for_WBFM.utils.projects.utils_data_subsets import write_data_subset_from_config
# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS

from DLC_for_WBFM.utils.projects.utils_filepaths import resolve_mounted_path_in_current_os

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd

ex = Experiment()
ex.add_config(project_path=None,
              DEBUG=False)

@ex.config
def cfg(project_path):
    # Manually load yaml files
    cfg = load_config(project_path)
    project_dir = Path(project_path).parent

    fname = Path(resolve_mounted_path_in_current_os(cfg['red_bigtiff_fname']))
    out_fname_red = fname.with_name(fname.name + "_preprocessed").with_suffix('.zarr')

    fname = Path(resolve_mounted_path_in_current_os(cfg['green_bigtiff_fname']))
    out_fname_green = fname.with_name(fname.name + "_preprocessed").with_suffix('.zarr')


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    opt = {'tiff_not_zarr': False,
           'pad_to_align_with_original': False,
           'DEBUG': _config['DEBUG']}
    cfg = _config['cfg']
    cfg['project_dir'] = _config['project_dir']
    cfg['project_path'] = _config['project_path']

    with safe_cd(_config['project_dir']):
        # print("WARNING: current preprocessing is not deterministic between channels if rigid correction is used")

        opt['out_fname'] = _config['out_fname_red']
        opt['save_fname_in_red_not_green'] = True
        write_data_subset_from_config(cfg, **opt)

        opt['out_fname'] = _config['out_fname_green']
        opt['save_fname_in_red_not_green'] = False
        write_data_subset_from_config(cfg, **opt)
