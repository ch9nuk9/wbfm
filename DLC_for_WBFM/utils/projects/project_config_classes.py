import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import pprint

from DLC_for_WBFM.utils.neuron_matching.class_frame_pair import FramePairOptions
from DLC_for_WBFM.utils.projects.physical_units import PhysicalUnitConversion
from DLC_for_WBFM.utils.projects.utils_filenames import check_exists, resolve_mounted_path_in_current_os
from DLC_for_WBFM.utils.projects.utils_project import load_config, safe_cd, edit_config, get_sequential_filename
from DLC_for_WBFM.utils.general.preprocessing.utils_tif import PreprocessingSettings


@dataclass
class ConfigFileWithProjectContext:
    self_path: str
    config: dict = None
    project_dir: str = None

    def resolve_relative_path_from_config(self, key) -> str:
        val = self.config.get(key, None)
        return self.resolve_relative_path(val)

    def resolve_relative_path(self, val: str) -> str:
        if val is None:
            return None
        relative_path = Path(self.project_dir).joinpath(val)
        return str(relative_path.resolve())

    def unresolve_absolute_path(self, val: str) -> str:
        if val is None:
            return None
        # NOTE: is_relative_to() only works for python >= 3.9
        # if Path(val).is_relative_to(self.project_dir):
        try:
            return str(Path(val).relative_to(self.project_dir))
        except ValueError:
            return val

    def update_on_disk(self):
        fname = self.resolve_relative_path(self.self_path)
        edit_config(fname, self.config)

    def to_json(self):
        return json.dumps(vars(self))

    def pickle_in_local_project(self, data, relative_path: str,
                                allow_overwrite=True, make_sequential_filename=False,
                                custom_writer=None,
                                **kwargs):
        """
        For objects larger than 4GB and python<3.8, protocol=4 must be specified directly

        https://stackoverflow.com/questions/29704139/pickle-in-python3-doesnt-work-for-large-data-saving
        """
        abs_path = self.resolve_relative_path(relative_path)
        if not abs_path.endswith('.pickle'):
            abs_path += ".pickle"
        if make_sequential_filename:
            abs_path = get_sequential_filename(abs_path)
        logging.info(f"Saving at: {relative_path}")
        check_exists(abs_path, allow_overwrite)
        if custom_writer:
            # Useful for pickling dataframes
            custom_writer(data, abs_path, **kwargs)
        else:
            with open(abs_path, 'wb') as f:
                pickle.dump(data, f, **kwargs)

    def h5_in_local_project(self, data: pd.DataFrame, relative_path: str,
                            allow_overwrite=True, make_sequential_filename=False, also_save_csv=False):
        abs_path = self.resolve_relative_path(relative_path)
        if not abs_path.endswith('.h5'):
            abs_path += ".h5"
        logging.info(f"Saving at: {relative_path}")
        check_exists(abs_path, allow_overwrite)
        if make_sequential_filename:
            abs_path = get_sequential_filename(abs_path)
        data.to_hdf(abs_path, key="df_with_missing")

        if also_save_csv:
            csv_fname = Path(abs_path).with_suffix('.csv')
            data.to_csv(csv_fname)

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

    def resolve_relative_path(self, val: str, prepend_subfolder=False) -> str:
        if val is None:
            return None

        if prepend_subfolder:
            final_path = os.path.join(self.project_dir, self.subfolder, val)
        else:
            final_path = os.path.join(self.project_dir, val)
        return str(Path(final_path).resolve())


@dataclass
class ModularProjectConfig(ConfigFileWithProjectContext):
    """
    Add functionality to get individual config files using the main project config filepath

    Returns config_file_with_project_context objects, instead of raw dictionaries for the subconfig files
    """

    def __post_init__(self):
        self.config = load_config(self.self_path)
        self.project_dir = str(Path(self.self_path).parent)

    def get_segmentation_config(self):
        fname = Path(self.config['subfolder_configs']['segmentation'])
        return SubfolderConfigFile(*self._check_path_and_load_config(fname))

    def get_training_config(self):
        fname = Path(self.config['subfolder_configs']['training_data'])
        return SubfolderConfigFile(*self._check_path_and_load_config(fname))

    def get_tracking_config(self):
        fname = Path(self.config['subfolder_configs']['tracking'])
        return SubfolderConfigFile(*self._check_path_and_load_config(fname))

    def get_preprocessing_config(self):
        """Different: plain dict, which is only used at the very beginning"""
        fname = Path(self.project_dir).joinpath('preprocessing_config.yaml')
        return PreprocessingSettings.load_from_yaml(fname)

    def get_traces_config(self):
        fname = Path(self.config['subfolder_configs']['traces'])
        return SubfolderConfigFile(*self._check_path_and_load_config(fname))

    def get_physical_unit_conversion_class(self) -> PhysicalUnitConversion:
        if 'physical_units' in self.config:
            return PhysicalUnitConversion(**self.config['physical_units'])
        else:
            logging.warning("Using default physical unit conversions")
            return PhysicalUnitConversion()

    def get_frame_pair_options(self, training_config=None) -> FramePairOptions:
        if training_config is None:
            training_config = self.get_training_config()
        pairwise_matches_params = training_config.config['pairwise_matching_params'].copy()
        pairwise_matches_params = FramePairOptions(**pairwise_matches_params)

        physical_units = self.config['physical_units']
        physical_unit_conversion = PhysicalUnitConversion(**physical_units)
        pairwise_matches_params.physical_unit_conversion = physical_unit_conversion

        return pairwise_matches_params

    def _check_path_and_load_config(self, subconfig_path: Path) -> Tuple[str, dict, str, str]:
        if subconfig_path.is_absolute():
            project_dir = subconfig_path.parent.parent
        else:
            project_dir = Path(self.self_path).parent
        with safe_cd(project_dir):
            cfg = load_config(subconfig_path)
        subfolder = subconfig_path.parent
        return str(subconfig_path), cfg, str(project_dir), str(subfolder)

    def get_log_dir(self):
        return str(Path(self.project_dir).joinpath('log'))

    def resolve_mounted_path_in_current_os(self, key):
        return Path(resolve_mounted_path_in_current_os(self.config[key]))


def synchronize_segment_config(project_path: str, segment_cfg: dict) -> dict:
    # For now, does NOT overwrite anything on disk
    project_cfg = load_config(project_path)

    if 'preprocessed_red' not in project_cfg:
        raise ValueError("Must preprocess data before the segmentation step")
    updates = {'video_path': project_cfg['preprocessed_red']}
    segment_cfg.update(updates)

    return segment_cfg


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
