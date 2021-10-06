import importlib
import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.projects.utils_filepaths import modular_project_config
from .global_vars_for_tests import project_path
import pytest


def _load_training_data() -> pd.DataFrame:
    cfg = modular_project_config(project_path)
    training_cfg = cfg.get_training_config()

    fname = training_cfg.resolve_relative_path_from_config('df_raw_3d_tracks')
    df_tracks = pd.read_hdf(fname)
    return df_tracks


def test_pipeline_step():
    # Run the sacred experiment from the actual script
    mod = importlib.import_module("DLC_for_WBFM.scripts.2-produce_training_data", package="DLC_for_WBFM")

    config_updates = {'project_path': project_path, 'DEBUG': False}

    mod.ex.run(config_updates=config_updates)


def test_saved_properly():
    df = _load_training_data()
    assert type(df) == pd.DataFrame


def test_finds_matches():
    cfg = modular_project_config(project_path)
    df = _load_training_data()

    expected_len = cfg.config['dataset_params']['num_frames'] - 1
    assert len(df) == expected_len

    minimum_expected_matches = 110
    assert df.shape[1] > minimum_expected_matches


@pytest.mark.xfail(reason="Known incorrect segmentation leads to unreasonable z changes")
def test_reasonable_z():
    df = _load_training_data()

    df_delta = df.diff()[1:]
    max_z_delta = 2
    all_neurons = list(df.columns.levels[0])

    for n in all_neurons:
        z_delta_series = df_delta[n]['z']
        assert all(np.abs(z_delta_series < max_z_delta))
