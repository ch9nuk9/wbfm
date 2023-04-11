import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import pprint

from wbfm.utils.external.utils_pandas import ensure_dense_dataframe
from wbfm.utils.general.utils_logging import setup_logger_object, setup_root_logger
from wbfm.utils.projects.utils_filenames import check_exists, resolve_mounted_path_in_current_os, \
    get_sequential_filename, get_location_of_new_project_defaults
from wbfm.utils.projects.utils_project import load_config, edit_config, safe_cd, update_project_config_path, \
    update_snakemake_config_path


@dataclass
class ConfigFileWithProjectContext:
    """
    Top-level configuration file class

    Knows how to:
    1. update itself on disk
    2. save new data inside the relevant project
    3. change filepaths between relative and absolute
    """

    self_path: str
    config: dict = None
    project_dir: str = None

    _logger: logging.Logger = None
    log_to_file: bool = True

    def __post_init__(self):
        self.config = load_config(self.self_path)
        self.project_dir = str(Path(self.self_path).parent)

    @property
    def logger(self):
        if self._logger is None:
            self.setup_logger('ConfigFile.log')
        return self._logger

    def setup_logger(self, relative_log_filename: str):
        log_filename = self.resolve_relative_path(os.path.join('log', relative_log_filename))
        self._logger = setup_logger_object(log_filename, self.log_to_file)
        return self._logger

    def setup_global_logger(self, relative_log_filename: str):
        log_filename = self.resolve_relative_path(os.path.join('log', relative_log_filename))
        logger = setup_root_logger(log_filename)
        return logger

    def update_self_on_disk(self):
        fname = self.resolve_relative_path(self.self_path)
        self.logger.info(f"Updating config file {fname} on disk")
        self.logger.debug(f"Updated values: {self.config}")
        try:
            edit_config(fname, self.config)
        except PermissionError as e:
            if Path(self.self_path).is_absolute():
                self.logger.debug(f"Skipped updating nonlocal file: {fname}")
            else:
                # Then it was a local file, and the error was real
                raise e

    def resolve_relative_path_from_config(self, key) -> str:
        val = self.config.get(key, None)
        return self.resolve_relative_path(val)

    def resolve_relative_path(self, val: str) -> Optional[str]:
        if val is None or Path(val).is_absolute():
            return val
        relative_path = Path(self.project_dir).joinpath(val)
        return str(relative_path.resolve())

    def unresolve_absolute_path(self, val: str) -> Optional[str]:
        if val is None:
            return val
        # NOTE: is_relative_to() only works for python >= 3.9
        # if Path(val).is_relative_to(self.project_dir):
        try:
            return str(Path(val).relative_to(self.project_dir))
        except ValueError:
            return val

    @property
    def absolute_self_path(self):
        return self.resolve_relative_path(self.self_path)

    def to_json(self):
        return json.dumps(vars(self))

    def pickle_data_in_local_project(self, data, relative_path: str,
                                     allow_overwrite=True, make_sequential_filename=False,
                                     custom_writer=None,
                                     **kwargs):
        """
        For objects larger than 4GB and python<3.8, protocol=4 must be specified directly

        https://stackoverflow.com/questions/29704139/pickle-in-python3-doesnt-work-for-large-data-saving
        """
        abs_path = self.resolve_relative_path(relative_path)
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        if not abs_path.endswith('.pickle'):
            abs_path += ".pickle"
        if make_sequential_filename:
            abs_path = get_sequential_filename(abs_path)
        self.logger.info(f"Saving at: {self.unresolve_absolute_path(abs_path)}")
        check_exists(abs_path, allow_overwrite)
        if custom_writer:
            # Useful for pickling dataframes
            custom_writer(data, abs_path, **kwargs)
        else:
            with open(abs_path, 'wb') as f:
                pickle.dump(data, f, **kwargs)

        return abs_path

    def h5_data_in_local_project(self, data: pd.DataFrame, relative_path: str,
                                 allow_overwrite=True, make_sequential_filename=False, also_save_csv=False):
        abs_path = self.resolve_relative_path(relative_path)
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        if not abs_path.endswith('.h5'):
            abs_path += ".h5"
        if make_sequential_filename:
            abs_path = get_sequential_filename(abs_path)
        self.logger.info(f"Saving at: {self.unresolve_absolute_path(abs_path)}")
        check_exists(abs_path, allow_overwrite)
        ensure_dense_dataframe(data).to_hdf(abs_path, key="df_with_missing")

        if also_save_csv:
            csv_fname = Path(abs_path).with_suffix('.csv')
            data.to_csv(csv_fname)

        return abs_path

    def __repr__(self):
        pp = pprint.PrettyPrinter(indent=2)
        return pp.pformat(self.config)


@dataclass
class SubfolderConfigFile(ConfigFileWithProjectContext):
    """
    Configuration file (loaded from .yaml) that knows the project it should be executed in

    In principle this config file is associated with a subfolder (and single step) of a project
    """

    subfolder: str = None

    @property
    def absolute_subfolder(self):
        return self.resolve_relative_path(self.subfolder)

    def __post_init__(self):
        pass

    def resolve_relative_path(self, raw_path: str, prepend_subfolder=False) -> Optional[str]:
        if raw_path is None:
            return None

        final_path = self._prepend_subfolder(raw_path, prepend_subfolder)
        return str(Path(final_path).resolve())

    def _prepend_subfolder(self, val, prepend_subfolder):
        if prepend_subfolder:
            final_path = os.path.join(self.project_dir, self.subfolder, val)
        else:
            final_path = os.path.join(self.project_dir, val)
        return final_path

    def h5_data_in_local_project(self, data: pd.DataFrame, relative_path: str, prepend_subfolder=False,
                                 **kwargs):
        path = self._prepend_subfolder(relative_path, prepend_subfolder)
        abs_path = super().h5_data_in_local_project(data, path, **kwargs)
        return abs_path


@dataclass
class ModularProjectConfig(ConfigFileWithProjectContext):
    """
    Add functionality to get individual config files using the main project config filepath

    Returns config_file_with_project_context objects, instead of raw dictionaries for the subconfig files

    Knows how to:
    1. find the individual config files of the substeps
    2. initialize the physical unit conversion class
    3. and loading other options classes

    """

    def get_segmentation_config(self) -> SubfolderConfigFile:
        fname = Path(self.config['subfolder_configs']['segmentation'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_training_config(self) -> SubfolderConfigFile:
        fname = Path(self.config['subfolder_configs']['training_data'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_tracking_config(self) -> SubfolderConfigFile:
        fname = Path(self.config['subfolder_configs']['tracking'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_preprocessing_config(self) -> ConfigFileWithProjectContext:
        """
        Not often used, except for updating the file.

        Note: NOT a subfolder
        """
        fname = self.get_preprocessing_config_filename()
        return ConfigFileWithProjectContext(fname)

    def get_preprocessing_config_filename(self):
        # In newer versions, it is in the dat folder and has an entry in the main config file
        fname = self.config['subfolder_configs'].get('preprocessing', None)
        fname = self.resolve_relative_path(fname)
        if fname is None or not Path(fname).exists():
            # In older versions, it was in the main folder
            fname = str(Path(self.project_dir).joinpath('preprocessing_config.yaml'))
        return fname

    def get_behavior_config(self) -> SubfolderConfigFile:
        fname = Path(self.project_dir).joinpath('behavior', 'behavior_config.yaml')
        if not fname.exists():
            # self.logger.warning("Project does not have a behavior config file")
            raise FileNotFoundError
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def get_traces_config(self) -> SubfolderConfigFile:
        fname = Path(self.config['subfolder_configs']['traces'])
        return SubfolderConfigFile(**self._check_path_and_load_config(fname))

    def _check_path_and_load_config(self, subconfig_path: Path,
                                    allow_config_to_not_exist: bool = False) -> Dict:
        if subconfig_path.is_absolute():
            project_dir = subconfig_path.parent.parent
        else:
            project_dir = Path(self.self_path).parent
        with safe_cd(project_dir):
            try:
                cfg = load_config(subconfig_path)
            except FileNotFoundError as e:
                if allow_config_to_not_exist:
                    cfg = dict()
                else:
                    raise e
        subfolder = subconfig_path.parent

        args = dict(self_path=str(subconfig_path),
                    config=cfg,
                    project_dir=str(project_dir),
                    _logger=self.logger,
                    subfolder=str(subfolder))
        return args

    def get_log_dir(self) -> str:
        foldername = Path(self.project_dir).joinpath('log')
        foldername.mkdir(exist_ok=True)
        return str(foldername)

    def _get_visualization_dir(self) -> str:
        foldername = Path(self.project_dir).joinpath('visualization')
        try:
            foldername.mkdir(exist_ok=True)
        except PermissionError:
            pass
        return str(foldername)

    def get_visualization_config(self, make_subfolder=False):
        fname = self.config['subfolder_configs'].get('visualization', None)
        if fname is None:
            # Assume the local folder is correct
            fname = os.path.join(self._get_visualization_dir(), 'visualization_config.yaml')
        fname = Path(fname)
        cfg = SubfolderConfigFile(**self._check_path_and_load_config(fname, allow_config_to_not_exist=True))
        if make_subfolder:
            try:
                Path(cfg.subfolder).mkdir(exist_ok=True)
            except PermissionError:
                pass
        return cfg

    def resolve_mounted_path_in_current_os(self, key) -> Optional[Path]:
        path = self.config.get(key, None)
        if path is None:
            return None
        return Path(resolve_mounted_path_in_current_os(path))

    def get_folder_with_calibration_for_day(self) -> Optional[Path]:
        raw_bigtiff_filename = self.resolve_mounted_path_in_current_os('red_bigtiff_fname')
        if raw_bigtiff_filename is None:
            raise FileNotFoundError(raw_bigtiff_filename)
        raw_bigtiff_filename = Path(raw_bigtiff_filename)
        folder_for_entire_day = raw_bigtiff_filename.parents[2]  # 3 folders up

        if not Path(folder_for_entire_day).exists():
            raise FileNotFoundError(folder_for_entire_day)
        return folder_for_entire_day

    def get_folder_with_background(self) -> Path:
        folder_for_entire_day = self.get_folder_with_calibration_for_day()
        folder_for_background = folder_for_entire_day.joinpath('background')
        if not folder_for_background.exists():
            raise FileNotFoundError(f"Could not find background folder {folder_for_background}")

        return folder_for_background

    def get_folder_with_calibration(self):
        folder_for_entire_day = self.get_folder_with_calibration_for_day()
        folder_for_calibration = folder_for_entire_day.joinpath('calibration')
        if not folder_for_calibration.exists():
            raise FileNotFoundError(f"Could not find calibration folder {folder_for_calibration}")

        return folder_for_calibration

    def get_folder_with_alignment(self):
        folder_for_entire_day = self.get_folder_with_calibration_for_day()
        folder_for_alignment = folder_for_entire_day.joinpath('alignment')
        if not folder_for_alignment.exists():
            raise FileNotFoundError(f"Could not find alignment folder {folder_for_alignment}")

        return folder_for_alignment

    def get_red_and_green_dot_alignment_bigtiffs(self) -> Tuple[Optional[str], Optional[str]]:
        folder_for_alignment = self.get_folder_with_alignment()

        red_btf_fname, green_btf_fname = None, None
        for subfolder in folder_for_alignment.iterdir():
            if subfolder.is_dir():
                if subfolder.name.endswith('alignment_Ch0'):
                    red_btf_fname = self._extract_btf_from_folder(subfolder)
                elif subfolder.name.endswith('alignment_Ch1'):
                    green_btf_fname = self._extract_btf_from_folder(subfolder)

        return red_btf_fname, green_btf_fname

    def get_red_and_green_grid_alignment_bigtiffs(self) -> Tuple[List[str], List[str]]:
        """
        Find bigtiffs for the grid pattern, for alignment. Expects 5 files, all with the pattern:
        {date}_alignment-3D-{location}_{channel}
        Example: /scratch/neurobiology/zimmer/ulises/wbfm/20220913/2022-09-13_11-55_alignment-3D-TopLeft_Ch0

        The locations are:
        center, TopLeft, TopRight, BottomRight, BottomLeft

        Returns a list of filenames, in this order

        """
        # Note: this will probably change in the future
        folder_for_entire_day = self.get_folder_with_calibration_for_day()
        folder_for_alignment = folder_for_entire_day.parent

        red_btf_fname, green_btf_fname = {}, {}
        prefix_list = ['center', 'TopLeft', 'TopRight', 'BottomRight', 'BottomLeft']

        for subfolder in folder_for_alignment.iterdir():
            if subfolder.is_dir():
                if subfolder.name.endswith('_Ch0'):
                    this_fname = self._extract_btf_from_folder(subfolder, allow_ome_tif=True)
                    try:
                        this_key = prefix_list[np.where([p in subfolder.name for p in prefix_list])[0][0]]
                        red_btf_fname[this_key] = this_fname
                    except IndexError:
                        # Not one of the ones we care about
                        pass
                elif subfolder.name.endswith('_Ch1'):
                    this_fname = self._extract_btf_from_folder(subfolder, allow_ome_tif=True)
                    try:
                        this_key = prefix_list[np.where([p in subfolder.name for p in prefix_list])[0][0]]
                        green_btf_fname[this_key] = this_fname
                    except IndexError:
                        # Not one of the ones we care about
                        pass

        red_btf_fname = [red_btf_fname[k] for k in prefix_list if red_btf_fname[k] is not None]
        green_btf_fname = [green_btf_fname[k] for k in prefix_list if green_btf_fname[k] is not None]
        if len(red_btf_fname) < len(prefix_list):
            logging.warning(f"Expected 5 alignment files, but only found {len(red_btf_fname)} : {red_btf_fname}")
        return red_btf_fname, green_btf_fname

    @staticmethod
    def _extract_btf_from_folder(subfolder, allow_ome_tif=False):
        btf_fname = None
        for file in subfolder.iterdir():
            if file.name.endswith('btf'):
                btf_fname = str(file)
            elif allow_ome_tif and file.name.endswith('ome.tif'):
                btf_fname = str(file)
        return btf_fname

    def get_behavior_raw_file_from_red_fname(self):
        """If the user did not set the behavior foldername, try to infer it from the red"""
        behavior_subfolder, flag = self.get_behavior_raw_parent_folder_from_red_fname()
        if not flag:
            return None
        # Second, get the file itself
        for content in behavior_subfolder.iterdir():
            if content.is_file():
                # UK spelling, and there may be preprocessed bigtiffs in the folder
                if str(content).endswith('-behaviour-bigtiff.btf'):
                    behavior_fname = behavior_subfolder.joinpath(content)
                    break
        else:
            print(f"Found no behavior file in {behavior_subfolder}, aborting")
            return None

        return behavior_fname, behavior_subfolder

    def get_behavior_raw_parent_folder_from_red_fname(self) -> Tuple[Optional[Path], bool]:
        # red_fname = self.config['red_bigtiff_fname']
        red_fname = self.resolve_mounted_path_in_current_os('red_bigtiff_fname')
        if red_fname is None:
            return None, False
        main_data_folder = Path(red_fname).parents[1]
        if not main_data_folder.exists():
            return None, False
        # First, get the subfolder
        for content in main_data_folder.iterdir():
            if content.is_dir():
                # Ulises uses UK spelling
                if 'behaviour' in content.name or 'BH' in content.name:
                    behavior_subfolder = main_data_folder.joinpath(content)
                    flag = True
                    break
        else:
            print(f"Found no behavior subfolder in {main_data_folder}, aborting")
            flag = False
            behavior_subfolder = None
        return behavior_subfolder, flag


def update_path_to_segmentation_in_config(cfg: ModularProjectConfig) -> SubfolderConfigFile:
    # For now, does NOT overwrite anything on disk

    segment_cfg = cfg.get_segmentation_config()
    train_cfg = cfg.get_training_config()

    metadata_path = segment_cfg.resolve_relative_path_from_config('output_metadata')
    # Add external detections
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Could not find external annotations")
    train_cfg.config['tracker_params']['external_detections'] = segment_cfg.unresolve_absolute_path(metadata_path)

    return train_cfg


def update_path_to_behavior_in_config(cfg: ModularProjectConfig):
    # Used to update old projects that were not initialized with a behavior folder
    try:
        _ = cfg.get_behavior_config()
        print("Project already has a behavior config; update cannot be clearly automated")
        return
    except FileNotFoundError:
        pass

    # Then make a new folder, file, and fill it
    fname = Path(cfg.project_dir).joinpath('behavior', 'behavior_config.yaml')
    fname.parent.mkdir(exist_ok=False)
    behavior_cfg = SubfolderConfigFile(self_path=str(fname), config={}, project_dir=cfg.project_dir, subfolder='behavior')

    # Fill variable 1: Try to find behavior annotations
    raw_behavior_foldername, flag = cfg.get_behavior_raw_parent_folder_from_red_fname()
    if not flag:
        cfg.logger.warning("Could not find behavior foldername, so will create empty config file")
    else:
        annotated_behavior_csb = raw_behavior_foldername.joinpath('beh_annotation.csv')
        if not annotated_behavior_csb.exists():
            cfg.logger.warning(f"Could not find behavior file {annotated_behavior_csb}, so will create empty config file")
        else:
            behavior_cfg.config['manual_behavior_annotation'] = str(annotated_behavior_csb)

    # After filling (or not), write to disk
    behavior_cfg.update_self_on_disk()


def rename_variable_in_config(project_path: str, vars_to_rename: dict):
    """
    Renames variables, especially for updating variable names

    Overwrites the config file on disk

    Parameters
    ----------
    project_path
    vars_to_rename - nested dict with following levels
        key0 = project_config name (None is main level)
            (project, preprocessing, segmentation, training, tracking, traces)
        key1 = Variable name. Can be nested. If this isn't found, then skip
        key2 = If vars_to_rename[key0][key1] is not a dict, then this is the new variable name
            else, then recurse

    Returns
    -------

    """

    cfg = ModularProjectConfig(project_path)

    def _update_config_value(_file_key, _cfg_to_update0, _new_name0, _old_name0):
        if _cfg_to_update0 is None:
            return

        if isinstance(_new_name0, dict):
            for _old_name1, _new_name1 in _new_name0.items():
                _cfg_to_update1 = _cfg_to_update0.get(_old_name1, None)
                _update_config_value(_file_key, _cfg_to_update1, _new_name0, _old_name0)
            return

        if _old_name0 not in _cfg_to_update0:
            msg = f"{_old_name0} not found in config {_file_key}, skipping"
            logging.warning(msg)
        else:
            new_val = _cfg_to_update0[_old_name0]
            if _new_name0 in _cfg_to_update0:
                msg = f"New name {_new_name0} already found in config {_file_key}!"
                raise NotImplementedError(msg)
            _cfg_to_update0[_new_name0] = new_val
        return _cfg_to_update0

    for file_key, vars_dict in vars_to_rename.items():
        if file_key == 'project':
            loaded_cfg = cfg
        else:
            # Note: this creates coupling with the naming convention...
            load_function_name = f'get_{file_key}_config'
            loaded_cfg = getattr(cfg, load_function_name)

        for old_name0, new_name0 in vars_dict.items():
            _cfg_to_update = loaded_cfg.config
            _update_config_value(file_key, _cfg_to_update, new_name0, old_name0)
        loaded_cfg.update_self_on_disk()


def make_project_like(project_path: str, target_directory: str,
                      steps_to_keep: list = None,
                      target_suffix: str = None,
                      new_project_name: str = None, verbose=1):
    """
    Copy all config files from a project, i.e. only the files that would exist in a new project

    Parameters
    ----------
    project_path - project to copy
    target_directory - parent folder within which to create the new project
    steps_to_keep - steps, if any, to keep absolute paths connecting to the old project.
        Should be the full name of the step, not just a number (and not including the number). Example:
        "steps_to_keep=['segmentation']"
    target_suffix - suffix for filename. Default is none
    new_project_name - optional new name for project. Default is same as old
    verbose

    Returns
    -------

    """

    assert project_path.endswith('.yaml'), f"Must pass a valid config file: {project_path}"
    assert os.path.exists(target_directory), f"Must pass a folder that exists: {target_directory}"

    project_dir = Path(project_path).parent
    if new_project_name is None:
        new_project_name = project_dir.name
    if target_suffix is not None:
        new_project_name = f"{new_project_name}{target_suffix}"
    target_project_name = Path(target_directory).joinpath(new_project_name)
    if os.path.exists(target_project_name):
        raise FileExistsError(f"There is already a project at: {target_project_name}")
    if verbose >= 1:
        print(f"Copying project {project_dir}")

    # Get a list of all files that should be present, relative to the project directory
    src = get_location_of_new_project_defaults()
    initial_fnames = list(Path(src).rglob('**/*'))
    if len(initial_fnames) == 0:
        print("Found no initial files, probably running this from the wrong directory")
        raise FileNotFoundError

    # Convert them to relative
    initial_fnames = {str(fname.relative_to(src)) for fname in initial_fnames}
    if verbose >= 3:
        print(f"Found initial files: {initial_fnames}")

    # Also get the filenames of the target folder
    target_fnames = list(Path(project_dir).rglob('**/*'))
    if verbose >= 3:
        print(f"Found target files: {target_fnames}")

    # Check each initial project fname, and if it is in the initial set, copy it
    for fname in target_fnames:
        if fname.is_dir():
            continue
        rel_fname = fname.relative_to(project_dir)
        new_fname = target_project_name.joinpath(rel_fname)
        if str(rel_fname) in initial_fnames:
            os.makedirs(new_fname.parent, exist_ok=True)
            shutil.copy(fname, new_fname)

            if verbose >= 1:
                print(f"Copying {rel_fname}")
        elif verbose >= 2:
            print(f"Not copying {rel_fname}")

    # Update the copied project config with the new dest folder
    update_project_config_path(target_project_name)

    # Connect the new project to old project config files, if any
    if steps_to_keep is not None:
        project_updates = dict(subfolder_configs=dict())
        old_cfg = ModularProjectConfig(project_path)
        all_steps = list(old_cfg.config['subfolder_configs'].keys())
        old_project_dir = old_cfg.project_dir

        for step in all_steps:
            subcfg_fname = old_cfg.config['subfolder_configs'].get(step, None)
            if subcfg_fname is None:
                raise NotImplementedError(step)

            if step in steps_to_keep:
                # Must make it absolute
                if Path(subcfg_fname).is_absolute():
                    project_updates['subfolder_configs'][step] = subcfg_fname
                else:
                    project_updates['subfolder_configs'][step] = os.path.join(old_project_dir, subcfg_fname)

            else:
                # Must explicitly include the relative path, otherwise it will be deleted
                if Path(subcfg_fname).is_absolute():
                    subcfg_fname = Path(subcfg_fname)
                    project_updates['subfolder_configs'][step] = os.path.join(subcfg_fname.parent.name, subcfg_fname.name)
                else:
                    project_updates['subfolder_configs'][step] = subcfg_fname

        dest_fname = 'project_config.yaml'
        project_fname = os.path.join(target_project_name, dest_fname)
        project_fname = str(Path(project_fname).resolve())
        edit_config(project_fname, project_updates)
    else:
        print("All new steps")

    # Also update the snakemake file with the project directory
    update_snakemake_config_path(target_project_name)
