"""
"""

import os
from pathlib import Path
import sacred
from wbfm.utils.general.preprocessing.bounding_boxes import calculate_bounding_boxes_from_fnames_and_save, \
    generate_legacy_bbox_fname
from sacred import Experiment
from sacred import SETTINGS
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.general.preprocessing.utils_preprocessing import PreprocessingSettings

from wbfm.pipeline.project_initialization import write_data_subset_using_config, zip_zarr_using_config, \
    calculate_number_of_volumes_from_tiff_file
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_filenames import generate_output_data_names
from wbfm.utils.projects.utils_project import safe_cd
import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, to_zip_zarr_using_7z=True, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_0b.log')
    project_dir = cfg.project_dir
    out_fname_red, out_fname_green = generate_output_data_names(cfg)
    preprocessing_cfg = cfg.get_preprocessing_config()

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
    num_frames = cfg.config['dataset_params']['num_frames']
    if num_frames is None:
        # Check the number of total frames in the video, and update the parameter
        # Note: requires correct value of num_slices
        num_raw_slices = _config['dataset_params']['num_slices']
        red_bigtiff_fname = cfg.config['red_bigtiff_fname']
        num_volumes = calculate_number_of_volumes_from_tiff_file(num_raw_slices, red_bigtiff_fname)
        num_frames = int(num_volumes)
        cfg.config['dataset_params']['num_frames'] = num_frames
        cfg.update_self_on_disk()

    logger = cfg.logger
    project_dir = _config['project_dir']
    red_output_fname = _config['out_fname_red']
    green_output_fname = _config['out_fname_green']
    preprocessing_cfg = _config['preprocessing_cfg']

    bbox_fname = preprocessing_cfg.config.get('bounding_boxes_fname', None)
    if bbox_fname is None:
        generate_legacy_bbox_fname(project_dir)

    with safe_cd(project_dir):
        preprocessing_settings = PreprocessingSettings.load_from_config(cfg)

        # Very first: calculate the alignment between the red and green channels (camera misalignment)
        preprocessing_settings.calculate_warp_mat(cfg)
        green_name = Path(green_output_fname)
        fname = green_name.parent / (green_name.stem + "_camera_alignment.pickle")
        preprocessing_settings.path_to_camera_alignment_matrix = fname

        # Second: within-stack alignment using the red channel, which will be saved to disk
        options['out_fname'] = red_output_fname
        options['save_fname_in_red_not_green'] = True
        # Location: same as the preprocessed red channel (possibly not the bigtiff)
        red_name = Path(options['out_fname'])
        fname = red_name.parent / (red_name.stem + "_preprocessed.pickle")
        preprocessing_settings.path_to_previous_warp_matrices = fname

        if Path(options['out_fname']).exists() and fname.exists():
            logger.info("Preprocessed red already exists; skipping to green")
        else:
            logger.info("Preprocessing red...")
            preprocessing_settings.do_mirroring = False
            assert preprocessing_settings.to_save_warp_matrices
            write_data_subset_using_config(cfg, preprocessing_settings=preprocessing_settings,
                                           which_channel='red', **options)

        # Third the green channel will read the warp matrices per-volume (step 2) and between cameras (step 1)
        logger.info("Preprocessing green...")
        options['out_fname'] = green_output_fname
        options['save_fname_in_red_not_green'] = False
        preprocessing_settings.to_use_previous_warp_matrices = True
        if cfg.config['dataset_params']['red_and_green_mirrored']:
            preprocessing_settings.do_mirroring = True
        write_data_subset_using_config(cfg, preprocessing_settings=preprocessing_settings,
                                       which_channel='green', **options)

        # Save the warp matrices (camera and per-volume) to disk if needed further
        preprocessing_settings.save_all_warp_matrices()

        # Also saving bounding boxes for future segmentation (speeds up and dramatically reduces false positives)
        video_fname = red_output_fname
        Path(bbox_fname).parent.mkdir(parents=True, exist_ok=True)
        calculate_bounding_boxes_from_fnames_and_save(video_fname, bbox_fname, num_frames)

        bbox_fname = preprocessing_cfg.unresolve_absolute_path(bbox_fname)
        preprocessing_cfg.config['bounding_boxes_fname'] = bbox_fname
        preprocessing_cfg.update_self_on_disk()

    if _config['to_zip_zarr_using_7z']:
        zip_zarr_using_config(cfg)

    logger.info("Finished.")
