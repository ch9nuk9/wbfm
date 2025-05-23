import logging
import os.path as osp
from tqdm.auto import tqdm

from wbfm.utils.external.utils_zarr import zarr_reader_folder_or_zipstore
from wbfm.utils.external.custom_errors import AnalysisOutOfOrderError, IncompleteConfigFileError
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project import safe_cd


def _check_and_print(all_to_check: list, description: str, verbose: int):
    all_exist = all(map(osp.exists, all_to_check))
    if verbose >= 1:
        if all_exist:
            if verbose >= 2:
                print(f"Found all files ({description})")
        else:
            print(f"Did not find some necessary files: {all_to_check}")
    return all_exist


def check_all_needed_data_for_step(project_config: ModularProjectConfig,
                                   step_index: int,
                                   raise_error=True,
                                   training_data_required=True,
                                   verbose=1):
    if project_config is None or not project_config.has_valid_self_path:
        logging.warning("No project config provided; cannot check data")
        if raise_error:
            raise IncompleteConfigFileError("No project config provided; cannot check data")
        else:
            return True
    flag = True
    if step_index > 0:
        flag = check_preprocessed_data(project_config, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Preprocessing')
    if step_index > 1:
        flag = check_segmentation(project_config, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Segmentation')
    if step_index > 2:
        if training_data_required:
            flag = check_training_final(project_config, verbose)
        else:
            flag = True
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Training data')
    if step_index > 3:
        flag = check_tracking(project_config, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Tracking')
    if step_index > 4:
        flag = check_traces(project_config, verbose)
        if not flag and raise_error:
            raise AnalysisOutOfOrderError('Traces')
    return flag


def check_preprocessed_data(project_config: ModularProjectConfig, verbose=0):

    try:
        p = project_config.get_preprocessing_class()
        all_to_check = [
            p.get_path_to_preprocessed_data(red_not_green=True),
            p.get_path_to_preprocessed_data(red_not_green=False)
        ]
        all_exist = _check_and_print(all_to_check, 'preprocessed data', verbose)

        return all_exist
    except (AssertionError, TypeError):
        return False


def check_segmentation(project_config: ModularProjectConfig, verbose=0):
    cfg_segment = project_config.get_segmentation_config()

    try:
        all_to_check = [
            cfg_segment.resolve_relative_path_from_config('output_masks'),
            cfg_segment.resolve_relative_path_from_config('output_metadata')
        ]
        all_exist = _check_and_print(all_to_check, 'segmentation', verbose)

        return all_exist
    except (AssertionError, TypeError):
        return False


def check_training_raw(project_config: ModularProjectConfig, verbose=0):
    cfg_training = project_config.get_training_config()

    try:
        with safe_cd(cfg_training.project_dir):
            training_folder = '2-training_data'
            file_names = ['clust_df_dat.pickle', 'frame_dat.pickle', 'match_dat.pickle']
            all_to_check = map(lambda file: osp.join(training_folder, 'raw', file), file_names)
            all_exist = _check_and_print(all_to_check, 'raw training data', verbose)
            return all_exist
    except (AssertionError, TypeError):
        return False


def check_training_only_tracklets(project_config: ModularProjectConfig, verbose=0):
    cfg_training = project_config.get_training_config()

    try:
        all_to_check = [
            cfg_training.resolve_relative_path_from_config('df_3d_tracklets'),
        ]
        all_exist = _check_and_print(all_to_check, 'all tracklets', verbose)
        return all_exist
    except (AssertionError, TypeError):
        return False


def check_training_final(project_config: ModularProjectConfig, verbose=0):
    cfg_training = project_config.get_training_config()

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


def check_tracking(project_config: ModularProjectConfig, verbose=0):
    tracking_cfg = project_config.get_tracking_config()

    try:
        all_to_check = [tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df')]
        all_exist = _check_and_print(all_to_check, 'tracking', verbose)

        return all_exist
    except (AssertionError, TypeError):
        return False


def check_traces(project_config: ModularProjectConfig, verbose=0):
    try:
        with safe_cd(project_config.project_dir):
            traces_cfg = project_config.get_traces_config()
            file_names = ['all_matches.pickle', 'green_traces.h5', 'red_traces.h5']
            # file_names = ['reindexed_masks.zarr.zip', 'all_matches.pickle', 'green_traces.h5', 'red_traces.h5']
            make_full_name = lambda file: traces_cfg.resolve_relative_path(file, prepend_subfolder=True)
            all_to_check = list(map(make_full_name, file_names))
            all_exist = _check_and_print(all_to_check, 'traces', verbose)

            return all_exist
    except (AssertionError, TypeError):
        return False


def check_zarr_file_integrity(project_config: ModularProjectConfig, verbose=0):
    p = project_config.get_preprocessing_class()
    fnames = [p.get_path_to_preprocessed_data(red_not_green=True),
              p.get_path_to_preprocessed_data(red_not_green=False)]

    for fname in fnames:
        logging.info(f"Checking integrity of {fname}")
        z = zarr_reader_folder_or_zipstore(fname)

        for frame in tqdm(z, leave=False):
            tmp = frame.shape


def print_sacred_log(project_config: str) -> None:
    from sacred.observers import TinyDbReader
    project_config = ModularProjectConfig(project_config)

    reader = TinyDbReader(project_config.get_log_dir())
    results = reader.fetch_report(indices=-1)

    try:
        print(results[0])
    except KeyError:
        print("Key error in the log; this means a step is in progress or the log is corrupted")


def get_project_status(project_config: ModularProjectConfig, verbose=2):
    """
    Returns the index of the last step that was completed

    Parameters
    ----------
    project_config
    verbose

    Returns
    -------

    """
    opt = dict(project_config=project_config, training_data_required=False, raise_error=False)

    project_config.logger.info("Determining status of project...")
    i_step = 1
    for i_step in tqdm([1, 2, 3, 4, 5]):
        passed = check_all_needed_data_for_step(step_index=i_step, **opt)
        if not passed:
            if verbose >= 1:
                project_config.logger.info(f"==============================")
                project_config.logger.info(f"Next pipeline step required: {i_step-1}")
                project_config.logger.info(f"==============================")
            break
    else:
        if verbose >= 1:
            project_config.logger.info("All steps of project are complete; manual annotation can begin")

    # Return last completed step
    return i_step - 1
