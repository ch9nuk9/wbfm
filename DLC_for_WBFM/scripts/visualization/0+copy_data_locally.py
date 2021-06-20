"""
"""

# main function
from DLC_for_WBFM.utils.projects.utils_data_subsets import write_data_subset_from_config
# Experiment tracking
import sacred
from sacred import Experiment

# Initialize sacred experiment
from DLC_for_WBFM.utils.projects.utils_project import load_config

ex = Experiment()
ex.add_config(project_path=None,
              out_fname=None,
              tiff_not_zarr=True,
              pad_to_align_with_original=False,
              do_only_training_data=False)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    cfg = load_config(_config['project_path'])
    tracking_cfg = load_config(cfg['subfolder_configs']['tracking'])
    if _config['do_only_training_data']:
        # Change config to match the frames used for training (assuming contiguous)
        cfg['dataset_params']['start_volume'] = tracking_cfg['training_data_3d']['which_frames'][0]
        cfg['dataset_params']['num_frames'] = tracking_cfg['training_data_3d']['num_training_frames']

    opt = {'out_fname': _config['out_fname'],
           'tiff_not_zarr': _config['tiff_not_zarr'],
           'pad_to_align_with_original': _config['pad_to_align_with_original']}
    write_data_subset_from_config(cfg, **opt)
