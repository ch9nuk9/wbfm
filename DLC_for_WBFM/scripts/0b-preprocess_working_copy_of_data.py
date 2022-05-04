"""
"""

import os
from pathlib import Path
import sacred
from DLC_for_WBFM.utils.general.preprocessing.bounding_boxes import calculate_bounding_boxes_from_fnames
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch
from DLC_for_WBFM.utils.general.preprocessing.utils_preprocessing import zip_zarr_using_config, PreprocessingSettings

from DLC_for_WBFM.utils.projects.utils_data_subsets import write_data_subset_from_config
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment()
ex.add_config(project_path=None, to_zip_zarr_using_7z=True, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    project_dir = cfg.project_dir

    fname = cfg.resolve_mounted_path_in_current_os('red_bigtiff_fname')
    out_fname_red = str(fname.with_name(fname.name + "_preprocessed").with_suffix('.zarr'))

    fname = cfg.resolve_mounted_path_in_current_os('green_bigtiff_fname')
    out_fname_green = str(fname.with_name(fname.name + "_preprocessed").with_suffix('.zarr'))

    bounding_box_fname = os.path.join(cfg.project_dir, '1-segmentation', 'bounding_boxes.pickle')
    segment_cfg = cfg.get_segmentation_config()

    fname = str(fname)  # For pickling in tinydb

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    options = {'tiff_not_zarr': False,
               'pad_to_align_with_original': False,
               'use_preprocessed_data': False,
               'DEBUG': _config['DEBUG']}
    cfg: ModularProjectConfig = _config['cfg']
    cfg.config['project_dir'] = _config['project_dir']
    cfg.config['project_path'] = _config['project_path']

    with safe_cd(_config['project_dir']):

        preprocessing_settings = PreprocessingSettings.load_from_config(cfg)
        # preprocessing_settings.find_background_files_from_raw_data_path(cfg)

        options['out_fname'] = _config['out_fname_red']
        options['save_fname_in_red_not_green'] = True
        # The preprocessing will be calculated based off the red channel, and will be saved to disk
        red_name = Path(options['out_fname'])
        fname = red_name.parent / (red_name.stem + "_preprocessed.pickle")
        preprocessing_settings.path_to_previous_warp_matrices = fname

        if not (Path(options['out_fname']).exists() and fname.exists()):
            print("Preprocessing red...")
            preprocessing_settings.do_mirroring = False
            assert preprocessing_settings.to_save_warp_matrices
            write_data_subset_from_config(cfg, preprocessing_settings=preprocessing_settings,
                                          which_channel='red', **options)
        else:
            print("Preprocessed red already exists; skipping to green")

        # Now the green channel will read the artifact as saved above
        print("Preprocessing green...")
        options['out_fname'] = _config['out_fname_green']
        options['save_fname_in_red_not_green'] = False
        preprocessing_settings.to_use_previous_warp_matrices = True
        if cfg.config['dataset_params']['red_and_green_mirrored']:
            preprocessing_settings.do_mirroring = True
        write_data_subset_from_config(cfg, preprocessing_settings=preprocessing_settings,
                                      which_channel='green', **options)

        # Save the warp matrices to disk if needed further
        preprocessing_settings.save_all_warp_matrices()

        # Also saving bounding boxes for future segmentation (speeds up and dramatically reduces false positives)
        video_fname = _config['out_fname_red']
        bbox_fname = _config['bounding_box_fname']
        calculate_bounding_boxes_from_fnames(video_fname, bbox_fname)

        segment_cfg = _config['segment_cfg']
        bbox_fname = segment_cfg.unresolve_absolute_path(bbox_fname)
        segment_cfg.config['bbox_fname'] = bbox_fname
        segment_cfg.update_self_on_disk()

    if _config['to_zip_zarr_using_7z']:
        zip_zarr_using_config(cfg)

    print("Finished.")

