"""
"""

# main function
import os
from pathlib import Path

from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
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

        preprocessing_fname = cfg['preprocessing_config']
        preprocessing_settings = PreprocessingSettings.load_from_yaml(preprocessing_fname)

        opt['out_fname'] = _config['out_fname_red']
        opt['save_fname_in_red_not_green'] = True
        # The preprocessing will be calculated based off the red channel, and will be saved to disk
        red_name = Path(opt['out_fname'])
        fname = red_name.parent / (red_name.stem + "_preprocessed.pickle")
        preprocessing_settings.path_to_previous_warp_matrices = fname
        assert preprocessing_settings.to_save_warp_matrices
        write_data_subset_from_config(cfg, preprocessing_settings=preprocessing_settings, **opt)

        # Now the green channel will read the artifact as saved above
        opt['out_fname'] = _config['out_fname_green']
        opt['save_fname_in_red_not_green'] = False
        preprocessing_settings.to_use_previous_warp_matrices = True
        write_data_subset_from_config(cfg, preprocessing_settings=preprocessing_settings, **opt)

        # Finalize by updating the preprocessing yaml
        preprocessing_settings.write_to_yaml(preprocessing_fname)
