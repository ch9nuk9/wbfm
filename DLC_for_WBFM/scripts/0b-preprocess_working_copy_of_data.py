"""
"""

# main function
from pathlib import Path

# Experiment tracking
import sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import TinyDbObserver
from DLC_for_WBFM.utils.external.monkeypatch_json import using_monkeypatch

from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings
from DLC_for_WBFM.utils.projects.utils_data_subsets import write_data_subset_from_config
from DLC_for_WBFM.utils.projects.utils_filepaths import resolve_mounted_path_in_current_os, modular_project_config
from DLC_for_WBFM.utils.projects.utils_project import safe_cd

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment

ex = Experiment()
ex.add_config(project_path=None,
              DEBUG=False)


@ex.config
def cfg(project_path):
    # Manually load yaml files
    cfg = modular_project_config(project_path)
    project_dir = cfg.project_dir

    fname = Path(resolve_mounted_path_in_current_os(cfg['red_bigtiff_fname']))
    out_fname_red = str(fname.with_name(fname.name + "_preprocessed").with_suffix('.zarr'))

    fname = Path(resolve_mounted_path_in_current_os(cfg['green_bigtiff_fname']))
    out_fname_green = str(fname.with_name(fname.name + "_preprocessed").with_suffix('.zarr'))

    fname = str(fname)  # For pickling in tinydb

    if not DEBUG:
        using_monkeypatch()
        log_dir = cfg.get_log_dir()
        ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    options = {'tiff_not_zarr': False,
               'pad_to_align_with_original': False,
               'use_preprocessed_data': False,
               'DEBUG': _config['DEBUG']}
    cfg = _config['cfg']
    cfg['project_dir'] = _config['project_dir']
    cfg['project_path'] = _config['project_path']

    with safe_cd(_config['project_dir']):

        preprocessing_fname = cfg['preprocessing_config']
        preprocessing_settings = PreprocessingSettings.load_from_yaml(preprocessing_fname)

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
            write_data_subset_from_config(cfg, preprocessing_settings=preprocessing_settings, **options)
        else:
            print("Preprocessed red already exists; skipping to green")

        # Now the green channel will read the artifact as saved above
        print("Preprocessing green...")
        options['out_fname'] = _config['out_fname_green']
        options['save_fname_in_red_not_green'] = False
        preprocessing_settings.to_use_previous_warp_matrices = True
        # preprocessing_settings.do_mirroring = False
        if cfg['dataset_params']['red_and_green_mirrored']:
            preprocessing_settings.do_mirroring = True
        write_data_subset_from_config(cfg, preprocessing_settings=preprocessing_settings, **options)

        # Save the warp matrices to disk if needed further
        preprocessing_settings.save_all_warp_matrices()

        print("Finished.")
