"""
"""

# main function
import os
from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS

from DLC_for_WBFM.utils.projects.utils_data_subsets import write_data_subset_from_config
from DLC_for_WBFM.utils.projects.utils_filepaths import resolve_mounted_path_in_current_os
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment()
ex.add_config(project_path=None,
              out_fname=None,
              tiff_not_zarr=False,
              pad_to_align_with_original=False,
              do_only_training_data=True,
              copy_locally=True,
              use_preprocessed_data=True,
              DEBUG=False)


@ex.config
def cfg(project_path, do_only_training_data, out_fname, copy_locally, tiff_not_zarr):
    # Manually load yaml files
    cfg = load_config(project_path)
    project_dir = Path(project_path).parent
    with safe_cd(project_dir):
        tracking_cfg = load_config(cfg['subfolder_configs']['tracking'])

    if do_only_training_data:
        # Change config to match the frames used for training (assuming contiguous)
        # TODO: Change to training_cfg
        cfg['dataset_params']['start_volume'] = tracking_cfg['training_data_3d']['which_frames'][0]
        cfg['dataset_params']['num_frames'] = tracking_cfg['training_data_3d']['num_training_frames']

        if out_fname is None:
            out_fname = os.path.join('2-training_data', 'training_data_red_channel.zarr')

    if not copy_locally:
        if out_fname is None:
            if tiff_not_zarr:
                raise ValueError("Did you really mean to re-copy the bigtiff as tiff?")
            else:
                fname = Path(resolve_mounted_path_in_current_os(cfg['red_bigtiff_fname']))
                out_fname = fname.with_name(fname.name + "_preprocessed").with_suffix('.zarr')


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    options = {'out_fname': _config['out_fname'],
               'tiff_not_zarr': _config['tiff_not_zarr'],
               'pad_to_align_with_original': _config['pad_to_align_with_original'],
               'use_preprocessed_data': _config['use_preprocessed_data'],
               'DEBUG': _config['DEBUG']}
    cfg = _config['cfg']
    cfg['project_dir'] = _config['project_dir']
    with safe_cd(_config['project_dir']):
        write_data_subset_from_config(cfg, **options)
