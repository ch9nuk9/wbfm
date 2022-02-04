import logging
import os.path as osp
from pathlib import Path

from DLC_for_WBFM.utils.general.custom_errors import AnalysisOutOfOrderError
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project import safe_cd, get_project_of_substep


def _check_and_print(all_to_check, description, verbose):
    all_exist = all(map(osp.exists, all_to_check))
    if verbose >= 1:
        if all_exist:
            print(f"Found all files ({description})")
        else:
            logging.warning(f"Did not find some necessary files: {all_to_check}")
    return all_exist


def check_all_needed_data_for_step(project_path, step_index: int,
                                   raise_error=True,
                                   training_data_required=True,
                                   verbose=0):
    if step_index > 0:
        flag = check_preprocessed_data(project_path, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Preprocessing')
    if step_index > 1:
        flag = check_segmentation(project_path, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Segmentation')
    if step_index > 2:
        if training_data_required:
            flag = check_training_final(project_path, verbose)
        else:
            flag = check_training_only_tracklets(project_path, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Training data')
    if step_index > 3:
        flag = check_tracking(project_path, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Tracking')
    if step_index > 4:
        flag = check_traces(project_path, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Traces')


def check_preprocessed_data(project_path, verbose=0):
    cfg = ModularProjectConfig(project_path)

    try:
        all_to_check = [
            cfg.config['preprocessed_red'],
            cfg.config['preprocessed_green']
        ]
        all_exist = _check_and_print(all_to_check, 'preprocessed data', verbose)

        return all_exist
    except (AssertionError, TypeError):
        return False


def check_segmentation(project_path, verbose=0):
    cfg = ModularProjectConfig(project_path)
    cfg_segment = cfg.get_segmentation_config()

    try:
        all_to_check = [
            cfg_segment.resolve_relative_path_from_config('output_masks'),
            cfg_segment.resolve_relative_path_from_config('output_metadata')
        ]
        all_exist = _check_and_print(all_to_check, 'segmentation', verbose)

        return all_exist
    except (AssertionError, TypeError):
        return False


def check_training_raw(project_path, verbose=0):
    cfg = ModularProjectConfig(project_path)
    cfg_training = cfg.get_training_config()

    try:
        with safe_cd(cfg_training.project_dir):
            training_folder = '2-training_data'
            file_names = ['clust_df_dat.pickle', 'frame_dat.pickle', 'match_dat.pickle']
            all_to_check = map(lambda file: osp.join(training_folder, 'raw', file), file_names)
            all_exist = _check_and_print(all_to_check, 'raw training data', verbose)
            return all_exist
    except (AssertionError, TypeError):
        return False


def check_training_only_tracklets(project_path, verbose=0):
    cfg = ModularProjectConfig(project_path)
    cfg_training = cfg.get_training_config()

    try:
        all_to_check = [
            cfg_training.resolve_relative_path_from_config('df_3d_tracklets'),
        ]
        all_exist = _check_and_print(all_to_check, 'final training data', verbose)
        return all_exist
    except (AssertionError, TypeError):
        return False


def check_training_final(project_path, verbose=0):
    cfg = ModularProjectConfig(project_path)
    cfg_training = cfg.get_training_config()

    try:
        all_to_check = [
            cfg_training.resolve_relative_path_from_config('df_3d_tracklets'),
            cfg_training.resolve_relative_path_from_config('df_training_3d_tracks'),
            cfg_training.resolve_relative_path_from_config('reindexed_masks'),
            cfg_training.resolve_relative_path_from_config('reindexed_metadata')
        ]
        all_exist = _check_and_print(all_to_check, 'final training data', verbose)
        return all_exist
    except (AssertionError, TypeError):
        return False


def check_tracking(project_path, verbose=0):
    cfg = ModularProjectConfig(project_path)
    tracking_cfg = cfg.get_tracking_config()

    try:
        all_to_check = [tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df')]
        all_exist = _check_and_print(all_to_check, 'tracking', verbose)

        return all_exist
    except (AssertionError, TypeError):
        return False


def check_traces(project_path, verbose=0):
    cfg = ModularProjectConfig(project_path)

    try:
        with safe_cd(cfg.project_dir):
            traces_cfg = cfg.get_traces_config()
            file_names = ['all_matches.pickle', 'green_traces.h5', 'red_traces.h5']
            make_full_name = lambda file: traces_cfg.resolve_relative_path(file, prepend_subfolder=True)
            all_to_check = map(make_full_name, file_names)
            all_exist = _check_and_print(all_to_check, 'traces', verbose)

            return all_exist
    except (AssertionError, TypeError):
        return False


def print_sacred_log(project_path: str) -> None:
    from sacred.observers import TinyDbReader

    cfg = ModularProjectConfig(project_path)

    reader = TinyDbReader(cfg.get_log_dir())
    results = reader.fetch_report(indices=-1)

    try:
        print(results[0])
    except KeyError:
        print("Key error in the log; this means a step is in progress or the log is corrupted")
