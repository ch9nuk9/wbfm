import importlib
from typing import Tuple

import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig
from DLC_for_WBFM.tests.global_vars_for_tests import project_path
import pytest


def _load_training_data() -> Tuple[object, ModularProjectConfig]:
    cfg = ModularProjectConfig(project_path)
    training_cfg = cfg.get_training_config()

    fname = training_cfg.resolve_relative_path_from_config('df_3d_tracklets')
    df_tracks = pd.read_hdf(fname)
    return df_tracks, cfg


def test_pipeline_step_a():
    mod = importlib.import_module("DLC_for_WBFM.scripts.2a-make_short_tracklets", package="DLC_for_WBFM")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_pipeline_step_b():
    mod = importlib.import_module("DLC_for_WBFM.scripts.2b-reindex_segmentation_training", package="DLC_for_WBFM")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_pipeline_step_c():
    mod = importlib.import_module("DLC_for_WBFM.scripts.2c-save_training_tracklets_as_dlc", package="DLC_for_WBFM")
    config_updates = {'project_path': project_path, 'DEBUG': False}
    mod.ex.run(config_updates=config_updates)


def test_saved_properly():
    df, _ = _load_training_data()
    assert type(df) == pd.DataFrame


def test_finds_matches():
    cfg = ModularProjectConfig(project_path)
    df, _ = _load_training_data()

    expected_len = cfg.config['dataset_params']['num_frames'] - 1
    assert len(df) == expected_len

    minimum_expected_matches = 110
    assert df.shape[1] > minimum_expected_matches


# @pytest.mark.xfail(reason="Known incorrect segmentation leads to unreasonable z changes")
def test_reasonable_z():
    df, cfg = _load_training_data()

    max_z_delta = cfg.get_training_config().config['postprocessing_params']['z_threshold']

    df_delta = df.diff()[1:]
    all_neurons = list(df.columns.levels[0])

    for n in all_neurons:
        z_delta_series = df_delta[n]['z']
        assert all(np.abs(z_delta_series <= max_z_delta))


# if __name__ == "__main__":
#     test_pipeline_step()
