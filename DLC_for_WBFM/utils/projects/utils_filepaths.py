import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Tuple

from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd, edit_config
from DLC_for_WBFM.utils.preprocessing.utils_tif import PreprocessingSettings


@dataclass
class config_file_with_project_context:
    """
    Configuration file (loaded from .yaml) that knows the project it should be executed in
    """

    config: dict
    project_dir: str

    def resolve_relative_path_from_config(self, key):
        val = self.config.get(key, None)
        return self.resolve_relative_path(val)

    def resolve_relative_path(self, val: str):
        return os.path.join(self.project_dir, val)

    def update_on_disk(self):
        with safe_cd(self.project_dir):
            edit_config(self.config['self_path'], self.config)

    def to_json(self):
        return json.dumps(vars(self))


@dataclass
class modular_project_config:
    """
    Add functionality to get individual config files using the main project config filepath

    Returns config_file_with_project_context objects, instead of raw dictionaries for the subconfig files
    """

    project_path: str
    config: dict = None
    project_dir: str = None

    def __post_init__(self):
        self.config = load_config(self.project_path)
        self.project_dir = str(Path(self.project_path).parent)

    def get_segmentation_config(self):
        fname = Path(self.config['subfolder_configs']['segmentation'])
        return config_file_with_project_context(*self._check_abs_and_load_config(fname))

    def get_training_config(self):
        fname = Path(self.config['subfolder_configs']['training_data'])
        return config_file_with_project_context(*self._check_abs_and_load_config(fname))

    def get_tracking_config(self):
        fname = Path(self.config['subfolder_configs']['tracking'])
        return config_file_with_project_context(*self._check_abs_and_load_config(fname))

    def get_preprocessing_config(self):
        """Different: plain dict, which is only used at the very beginning"""
        fname = Path(self.project_dir).joinpath('preprocessing_config.yaml')
        return PreprocessingSettings.load_from_yaml(fname)

    def get_traces_config(self):
        fname = Path(self.config['subfolder_configs']['traces'])
        return config_file_with_project_context(*self._check_abs_and_load_config(fname))

    def _check_abs_and_load_config(self, fname: Path) -> Tuple[dict, str]:
        if fname.is_absolute():
            project_dir = fname.parent.parent
        else:
            project_dir = Path(self.project_path).parent
        with safe_cd(project_dir):
            cfg = load_config(fname)
        return cfg, str(project_dir)

    def to_json(self):
        return json.dumps(vars(self))

    def get_log_dir(self):
        return str(Path(self.project_dir).joinpath('log'))

    def save_in_local_project(self, data, relative_path):
        abs_path = self.resolve_path_relative_to_project(relative_path)
        with open(abs_path, 'wb') as f:
            pickle.dump(data, f)

    def resolve_path_relative_to_project(self, relative_path):
        return Path(self.project_path).joinpath(relative_path)

    def resolve_mounted_path_in_current_os(self, key):
        return Path(resolve_mounted_path_in_current_os(self.config[key]))


def resolve_mounted_path_in_current_os(path: str, verbose: int = 1) -> str:
    """
    Removes windows-specific mounted drive names (Y:, D:, etc.) and replaces them with the networked system equivalent

    Does nothing if the path is relative

    Note: This is specific to the Zimmer lab, as of 23.06.2021 (at the IMP)
    """
    is_abs = PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()
    if not is_abs:
        return path

    if verbose >= 1:
        print(f"Checking path {path} on os {os.name}...")

    # Swap mounted drive locations
    # UPDATE REGULARLY
    mounted_drive_dict = {
        'Y:': "/groups/zimmer"
    }

    for win_drive, linux_drive in mounted_drive_dict.items():
        is_linux = "ix" in os.name.lower()
        is_windows_style = path.startswith(win_drive)
        is_windows = os.name.lower() == "windows" or os.name.lower() == "nt"
        is_linux_style = path.startswith(linux_drive)

        if is_linux and is_windows_style:
            path = path.replace(win_drive, linux_drive)
            path = str(Path(path).resolve())
        if is_windows and is_linux_style:
            path = path.replace(linux_drive, win_drive)
            path = str(Path(path).resolve())

    # Check for unreachable local drives
    local_drives = ['C:', 'D:']
    if "ix" in os.name.lower():
        for drive in local_drives:
            if path.startswith(drive):
                raise FileNotFoundError("File mounted to local drive; network system can't find it")

    if verbose >= 1:
        print(f"Resolved path to {path}")
    return path


def synchronize_segment_config(project_path: str, segment_cfg: dict) -> dict:
    # For now, does NOT overwrite anything on disk
    project_cfg = load_config(project_path)

    if 'preprocessed_red' not in project_cfg:
        raise ValueError("Must preprocess data before the segmentation step")
    updates = {'video_path': project_cfg['preprocessed_red']}
    segment_cfg.update(updates)

    # segment_folder = get_absname(project_path, 'segmentation')
    # updates = {'output_folder': segment_folder}
    # segment_cfg['output_params'].update(updates)

    return segment_cfg


def update_path_to_segmentation_in_config(cfg: modular_project_config) -> config_file_with_project_context:
    # For now, does NOT overwrite anything on disk

    segment_cfg = cfg.get_segmentation_config()
    train_cfg = cfg.get_training_config()

    metadata_path = segment_cfg.resolve_relative_path_from_config('output_metadata')
    # Add external detections
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Could not find external annotations")
    train_cfg.config['tracker_params']['external_detections'] = metadata_path

    return train_cfg
