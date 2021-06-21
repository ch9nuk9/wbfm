"""
"""

# main function
from pathlib import Path

from DLC_for_WBFM.utils.projects.utils_data_subsets import write_data_subset_from_config
# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd

ex = Experiment()
ex.add_config(project_path=None,
              out_fname=None,
              tiff_not_zarr=True,
              pad_to_align_with_original=False,
              do_only_training_data=False)

@ex.config
def cfg(project_path, do_only_training_data):
    # Manually load yaml files
    cfg = load_config(project_path)
    project_dir = Path(project_path).parent
    with safe_cd(project_dir):
        tracking_cfg = load_config(cfg['subfolder_configs']['tracking'])

    if do_only_training_data:
        # Change config to match the frames used for training (assuming contiguous)
        cfg['dataset_params']['start_volume'] = tracking_cfg['training_data_3d']['which_frames'][0]
        cfg['dataset_params']['num_frames'] = tracking_cfg['training_data_3d']['num_training_frames']


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    opt = {'out_fname': _config['out_fname'],
           'tiff_not_zarr': _config['tiff_not_zarr'],
           'pad_to_align_with_original': _config['pad_to_align_with_original']}
    cfg = _config['cfg']
    cfg['project_dir'] = _config['project_dir']
    with safe_cd(_config['project_dir']):
        write_data_subset_from_config(cfg, **opt)
